#  ___________________________________________________________________________
#
#  EGRET: Electrical Grid Research and Engineering Tools
#  Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
This module contains several helper functions and classes that are useful when
modifying the data dictionary
"""
import os, shutil, glob, json, gc
import egret.model_library.transmission.tx_utils as tx_utils
import numpy as np
import pandas as pd
from egret.data.lopf_utils import get_solution_file_location
from egret.models.acpf import create_psv_acpf_model, solve_acpf
from egret.common.log import logger
from math import sqrt


def update_solution_dicts(md, name="model_name", solution_dict=None):
    sd = solution_dict
    if sd is None:
        sd = dict()
        sd['pg'] = dict()
        sd['qg'] = dict()
        sd['pf'] = dict()
        sd['pfl'] = dict()
        sd['qf'] = dict()
        sd['qfl'] = dict()
        sd['va'] = dict()
        sd['vm'] = dict()

    gen = md.attributes(element_type='generator')
    bus = md.attributes(element_type='bus')
    branch = md.attributes(element_type='branch')
    system_data = md.data['system']

    sd['pg'].update({name: gen['pg']})
    sd['qg'].update({name: gen['qg']})
    sd['pf'].update({name: branch['pf']})
    sd['pfl'].update({name: branch['pfl']})
    sd['qf'].update({name: branch['qf']})
    sd['qfl'].update({name: branch['qfl']})
    sd['va'].update({name: bus['va']})
    sd['vm'].update({name: bus['vm']})

    return sd

def display_solution_dicts(sd, key1=None, key2=None, N=10):

    for k,d in sd.items():
        cols = list(d.keys())
        if key1 not in cols and key2 not in cols:
            key1 = cols[0]
            key2 = cols[1]

        dd = {key1:d[key1], key2:d[key2]}
        df = pd.DataFrame.from_dict(dd)
        df['abs_diff'] = abs(df[key1] - df[key2])
        print('...{}:'.format(k))
        print(df.sort_values('abs_diff',ascending=False).head(N))


def termination_condition(md):

    val = md.data['results']['termination']

    return val

def optimal(md):

    tc = md.data['results']['termination']
    if tc == 'optimal':
        return 1
    return 0

def infeasible(md):

    tc = md.data['results']['termination']
    if tc == 'infeasible':
        return 1
    return 0

def maxTimeLimit(md):

    tc = md.data['results']['termination']
    if tc == 'maxTimeLimit':
        return 1
    return 0

def maxIterations(md):

    tc = md.data['results']['termination']
    if tc == 'maxIterations':
        return 1
    return 0

def solverFailure(md):

    tc = md.data['results']['termination']
    if tc == 'solverFailure':
        return 1
    return 0

def internalSolverError(md):

    tc = md.data['results']['termination']
    if tc == 'internalSolverError':
        return 1
    return 0

def duals(md):
    try:
        val = md.data['results']['duals']
        return val
    except KeyError as e:
        logger.info('...ModelData is missing: {}'.format(str(e)))
        return 0

def solve_time(md):

    if not optimal(md):
        return None
    val = md.data['results']['time']
    return val

def num_buses(md):

    bus_attrs = md.attributes(element_type='bus')
    val = len(bus_attrs['names'])

    return val


def num_branches(md):

    branch_attrs = md.attributes(element_type='branch')
    val = len(branch_attrs['names'])

    return val


def num_constraints(md):

    if not optimal(md):
        return None
    val = md.data['results']['#_cons']
    return val

def num_variables(md):

    if not optimal(md):
        return None
    val = md.data['results']['#_vars']
    return val

def num_nonzeros(md):

    if not optimal(md):
        return None
    results = md.data['results']

    if '#_nz' in results:
        val = results['#_nz']
        return val

    return None

def model_density(md):

    if not optimal(md):
        return None
    results = md.data['results']

    if '#_nz' in results:
        nc = results['#_cons']
        nv = results['#_vars']
        nz = results['#_nz']
        val = nz / ( nc * nv )
        return val

    return None


def total_cost(md):

    if not optimal(md):
        return None
    val = md.data['system']['total_cost']

    return val

def lmp(md):

    if not duals(md):
        return None

    buses = dict(md.elements(element_type='bus'))
    lmp = {}

    for b, bus in buses.items():
        lmp[b] = bus['lmp']

    return lmp

def lmp_error(md):

    from egret.data.model_data import ModelData

    # get ACOPF filename for same multiplier
    mult = md.data['system']['mult']
    filename = md.data['system']['model_name']
    filename += '_acopf_'
    filename += '{0:04.0f}.json'.format(mult * 1000)

    case_folder = get_solution_file_location(md.data['system']['model_name'])
    dict_ac = json.load(open(os.path.join(case_folder, filename)))
    md_ac = ModelData(dict_ac)
    lmp_ac = lmp(md_ac)
    lmp_md = lmp(md)

    buses = list(set([k for k in lmp_ac.keys()] + [k for k in lmp_md.keys()]))
    lmp_error = {}
    for b in buses:
        try:
            lmp_error[b] = lmp_md[b] - lmp_ac[b]
        except KeyError:
            lmp_error[b] = None

    return lmp_error

def ploss(md):

    if not optimal(md):
        return None
    val = md.data['system']['ploss']

    return val

def qloss(md):

    if not optimal(md):
        return None
    val = md.data['system']['qloss']

    return val

def pgen(md):

    if not optimal(md):
        return None
    gens = dict(md.elements(element_type='generator'))
    dispatch = {}

    for g,gen in gens.items():
        dispatch[g] = gen['pg']

    return dispatch

def qgen(md):

    if not optimal(md):
        return None
    gens = dict(md.elements(element_type='generator'))
    dispatch = {}

    for g,gen in gens.items():
        dispatch[g] = gen['qg']

    return dispatch

def pflow(md):

    if not optimal(md):
        return None
    branches = dict(md.elements(element_type='branch'))
    flow = {}

    for b,branch in branches.items():
        flow[b] = branch['pf']

    return flow

def qflow(md):

    if not optimal(md):
        return None
    branches = dict(md.elements(element_type='branch'))
    flow = {}

    for b,branch in branches.items():
        flow[b] = branch['qf']

    return flow

def vmag(md):

    if not optimal(md):
        return None
    buses = dict(md.elements(element_type='bus'))
    vm = {}

    for b,bus in buses.items():
        vm[b] = bus['vm']

    return vm


def solve_infeas_model(model_data):

    # initial reference bus dispatch
    lin_gens = dict(model_data.elements(element_type='generator'))
    lin_buses = dict(model_data.elements(element_type='bus'))
    lin_branches = dict(model_data.elements(element_type='branch'))
    gens_by_bus = tx_utils.gens_by_bus(lin_buses, lin_gens)
    ref_bus = model_data.data['system']['reference_bus']
    slack_p_init = sum(lin_gens[gen_name]['pg'] for gen_name in gens_by_bus[ref_bus])

    def empty_acpf_dict(termination=None):
        acpf_dict = {}
        acpf_dict['acpf_termination'] = termination
        acpf_dict['time'] = None
        acpf_dict['balance_slack'] = None
        acpf_dict['acpf_slack'] = None
        acpf_dict['vm_viol'] = None
        acpf_dict['thermal_viol'] = None
        acpf_dict['pf_error'] = None
        acpf_dict['qf_error'] = None
        return acpf_dict

    if model_data.data['results']['termination'] != 'optimal':
        return empty_acpf_dict(termination='skipped')

    if 'filename' in list(model_data.data['system'].keys()):
        is_acopf = 'acopf' in model_data.data['system']['filename']
    else:
        is_acopf = False

    # solve ACPF or return empty results and print exception message
    try:
        if 'filename' in model_data.data['system'].keys():
            logger.critical('>>>> Solving ACPF for {}'.format(model_data.data['system']['filename']))
        kwargs = {}
        kwargs['include_feasibility_slack'] = True
        kwargs['timelimit'] = 1000
        md, results = solve_acpf(model_data, "ipopt", return_results=True, return_model=False, solver_tee=False, **kwargs)
        termination = results.solver.termination_condition.__str__()
        balance_slack = md.data['system']['balance_slack']
        logger.critical('<<<< ACPF terminated {} with {} power balance slack.'.format(termination, balance_slack))
    except Exception as e:
        message = str(e)
        logger.critical('...EXCEPTION OCCURRED: {}'.format(message))
        if 'infeasible' in message:
            return empty_acpf_dict(termination='infeasible')
        else:
            raise e

    vm_viol_dict = dict()
    thermal_viol_dict = dict()

    AC_gens = dict(md.elements(element_type='generator'))
    AC_buses = dict(md.elements(element_type='bus'))
    AC_branches = dict(md.elements(element_type='branch'))
    gens_by_bus = tx_utils.gens_by_bus(AC_buses, AC_gens)

    # calculate change in slackbus P dispatch
    ref_bus = md.data['system']['reference_bus']
    slack_p_acpf = sum(AC_gens[gen_name]['pg'] for gen_name in gens_by_bus[ref_bus])
    slack_p = slack_p_acpf - slack_p_init

    # calculate voltage infeasibilities
    for bus_name, bus_dict in AC_buses.items():
        if bus_dict['v_max'] is None or bus_dict['v_min'] is None:
            continue
        vm = bus_dict['vm']
        if vm > bus_dict['v_max']:
            vm_viol_dict[bus_name] = vm - bus_dict['v_max']
        elif vm < bus_dict['v_min']:
            vm_viol_dict[bus_name] = vm - bus_dict['v_min']
        else:
            vm_viol_dict[bus_name] = 0

    # calculate thermal infeasibilities
    for branch_name, branch_dict in AC_branches.items():
        if branch_dict['rating_long_term'] is None:
            continue
        sf = sqrt(branch_dict["pf"]**2 + branch_dict["qf"]**2)
        st = sqrt(branch_dict["pt"]**2 + branch_dict["qt"]**2)
        if sf > st: # to avoid double counting
            if sf > branch_dict['rating_long_term']:
                thermal_viol_dict[branch_name] = sf - branch_dict['rating_long_term']
            else:
                thermal_viol_dict[branch_name] = 0
        elif st > branch_dict['rating_long_term']:
            thermal_viol_dict[branch_name] = st - branch_dict['rating_long_term']
        else:
            thermal_viol_dict[branch_name] = 0

    # calculate flow errors
    pf_error = {}
    qf_error = {}
    has_qf = not any([branch['qf'] is None for bn,branch in lin_branches.items() if 'qf' in branch.keys()])
    for k, branch in lin_branches.items():
        if is_acopf:
            pf_ac = AC_branches[k]['pf']
            qf_ac = AC_branches[k]['qf']
        else:
            pf_ac = (AC_branches[k]['pf'] - AC_branches[k]['pt']) / 2
            qf_ac = (AC_branches[k]['qf'] - AC_branches[k]['qt']) / 2
        pf_error[k] = branch['pf'] - pf_ac
        if has_qf:
            qf_error[k] = branch['qf'] - qf_ac

    if not has_qf:
        qf_error = None

    del md
    gc.collect()

    acpf_dict = {}
    acpf_dict['acpf_termination'] = termination
    acpf_dict['time'] = results.Solver.Time
    acpf_dict['balance_slack'] = balance_slack
    acpf_dict['acpf_slack'] = slack_p
    acpf_dict['vm_viol'] = vm_viol_dict
    acpf_dict['thermal_viol'] = thermal_viol_dict
    acpf_dict['pf_error'] = pf_error
    acpf_dict['qf_error'] = qf_error

    return acpf_dict

def get_acpf_data(md, key='acpf_slack', overwrite_existing=False):

    # repopulate data if not in the ModelData or an ovewrite is desired
    if 'acpf_data' not in md.data.keys() or overwrite_existing:
        repopulate_acpf_to_modeldata(md)

    acpf_data = md.data['acpf_data']
    if not acpf_data['acpf_termination'] == 'optimal':
        return None
    elif key in acpf_data.keys():
        return acpf_data[key]
    else:
        return None

def repopulate_acpf_to_modeldata(md, abs_tol_vm=1e-6, rel_tol_therm=0.01, write_to_json=True):

    acpf_data = solve_infeas_model(md)

    bus_attrs = md.attributes(element_type='bus')
    branch_attrs = md.attributes(element_type='branch')
    num_bus = len(bus_attrs['names'])
    num_branch = len(branch_attrs['names'])
    thermal_max = branch_attrs['rating_long_term']

    vm_viol = acpf_data['vm_viol']
    if bool(vm_viol):
        vm_viol_pos = [v for v in vm_viol.values()]
        if bool(vm_viol_pos):
            acpf_data['sum_vm_UB_viol'] = sum(vm_viol_pos)
            acpf_data['max_vm_UB_viol'] = max(vm_viol_pos)
            acpf_data['avg_vm_UB_viol'] = acpf_data['sum_vm_UB_viol'] / len(vm_viol_pos)
            acpf_data['pct_vm_UB_viol'] = len([v for v in vm_viol.values() if v > abs_tol_vm]) / num_bus

        vm_viol_neg = [v for k,v in vm_viol.items()]
        if bool(vm_viol_neg):
            acpf_data['sum_vm_LB_viol'] = sum(vm_viol_neg)
            acpf_data['max_vm_LB_viol'] = max(vm_viol_neg)
            acpf_data['avg_vm_LB_viol'] = acpf_data['sum_vm_LB_viol'] / len(vm_viol_neg)
            acpf_data['pct_vm_LB_viol'] = len([v for k,v in vm_viol.items() if v < -abs_tol_vm]) / num_bus

        vm_viol_list = vm_viol_pos + vm_viol_neg
        if bool(vm_viol_list):
            acpf_data['sum_vm_viol'] = sum(vm_viol_list)
            acpf_data['max_vm_viol'] = max(vm_viol_list)
            acpf_data['avg_vm_viol'] = acpf_data['sum_vm_viol'] / len(vm_viol_list)
            acpf_data['pct_vm_viol'] = len([v for v in vm_viol_list if abs(v) > abs_tol_vm]) / num_bus

    thermal_viol = acpf_data['thermal_viol']
    if bool(thermal_viol):
        thermal_viol_list = [v for k,v in thermal_viol.items()]
        acpf_data['sum_thermal_viol'] = sum(thermal_viol_list)
        acpf_data['max_thermal_viol'] = max(thermal_viol_list)
        acpf_data['avg_thermal_viol'] = acpf_data['sum_thermal_viol'] / len(thermal_viol_list)
        acpf_data['pct_thermal_viol'] = len([v for k,v in thermal_viol.items() if v > rel_tol_therm * thermal_max[k]]) / num_branch

    pf_error = acpf_data['pf_error']
    if bool(pf_error):
        pf_error_list = [v for v in pf_error.values()]
        acpf_data['pf_error_1_norm'] = np.linalg.norm(pf_error_list, ord=1)
        acpf_data['pf_error_inf_norm'] = np.linalg.norm(pf_error_list, ord=np.inf)

    qf_error = acpf_data['qf_error']
    if bool(qf_error):
        qf_error_list = [v for v in qf_error.values()]
        acpf_data['qf_error_1_norm'] = np.linalg.norm(qf_error_list, ord=1)
        acpf_data['qf_error_inf_norm'] = np.linalg.norm(qf_error_list, ord=np.inf)

    # save acpf_data to JSON
    md.data['acpf_data'] = acpf_data
    system_data = md.data['system']
    if write_to_json and 'filename' in system_data.keys():
        #data_utils_deprecated.destroy_dicts_of_fdf(md)
        filename = system_data['filename']
        model_name = system_data['model_name']
        md.write_to_json(filename)
        save_to_solution_directory(filename, model_name)
    elif not write_to_json:
        print('Did not write to JSON.')
    else:
        print([system_data.keys()])
        print('Failed to write modelData to json.')


def save_to_solution_directory(filename, model_name):

    # directory locations
    cwd = os.getcwd()
    source = os.path.join(cwd, filename + '.json')
    destination = get_solution_file_location(model_name)

    if not os.path.exists(destination):
        os.makedirs(destination)

    if not glob.glob(source):
        print('No files to move.')
    else:
        #print('saving to dest: {}'.format(destination))

        for src in glob.glob(source):
            #print('src:  {}'.format(src))
            folder, file = os.path.split(src)
            dest = os.path.join(destination, file) # full destination path will overwrite existing files
            shutil.move(src, dest)

    return destination


def vm_UB_viol_sum(md):
    '''
    Returns the sum of voltage upper bound infeasibilites
    Note: returned value is in p.u.
    '''

    if not optimal(md):
        return None
    sum_vm_UB_viol = get_acpf_data(md, key='sum_vm_UB_viol')

    return sum_vm_UB_viol


def vm_LB_viol_sum(md):
    '''
    Returns the sum of voltage lower bound infeasibilities
    Note: returned value is in p.u.
    '''

    if not optimal(md):
        return None
    sum_vm_LB_viol = get_acpf_data(md, key='sum_vm_LB_viol')

    return sum_vm_LB_viol


def vm_viol_sum(md):
    '''
    Returns the sum of all voltage infeasibilities
    Note: returned value is in p.u.
    '''

    if not optimal(md):
        return None
    sum_vm_viol = get_acpf_data(md, key='sum_vm_viol')

    return sum_vm_viol


def thermal_viol_sum(md):
    '''
    Returns the sum of thermal limit infeasibilites
    Note: returned value is in MVA
    '''

    if not optimal(md):
        return None
    sum_thermal_viol = get_acpf_data(md, key='sum_thermal_viol')

    return sum_thermal_viol


def vm_UB_viol_avg(md):
    '''
    Returns the average of voltage upper bound infeasibilites
    Note: returned value is in p.u.
    '''

    if not optimal(md):
        return None
    avg_vm_UB_viol = get_acpf_data(md, key='avg_vm_UB_viol')

    return avg_vm_UB_viol

def vm_LB_viol_avg(md):
    '''
    Returns the average of voltage lower bound infeasibilities
    Note: returned value is in p.u.
    '''

    if not optimal(md):
        return None
    avg_vm_LB_viol = get_acpf_data(md, key='avg_vm_LB_viol')

    return avg_vm_LB_viol


def vm_viol_avg(md):
    '''
    Returns the average of all voltage infeasibilities
    Note: returned value is in p.u.
    '''

    if not optimal(md):
        return None
    avg_vm_viol = get_acpf_data(md, key='avg_vm_viol')

    return avg_vm_viol


def thermal_viol_avg(md):
    '''
    Returns the sum of thermal limit infeasibilites
    Note: returned value is in MVA
    '''

    if not optimal(md):
        return None
    avg_thermal_viol = get_acpf_data(md, key='avg_thermal_viol')

    return avg_thermal_viol


def vm_UB_viol_max(md):
    '''
    Returns the maximum of voltage upper bound infeasibilites
    Note: returned value is in p.u.
    '''

    if not optimal(md):
        return None
    max_vm_UB_viol = get_acpf_data(md, key='max_vm_UB_viol')

    return max_vm_UB_viol


def vm_LB_viol_max(md):
    '''
    Returns the maximum of voltage lower bound infeasibilities
    Note: returned value is in p.u.
    '''

    if not optimal(md):
        return None
    max_vm_LB_viol = get_acpf_data(md, key='max_vm_LB_viol')

    return max_vm_LB_viol


def vm_viol_max(md):
    '''
    Returns the maximum of all voltage infeasibilities
    Note: returned value is in p.u.
    '''

    if not optimal(md):
        return None
    max_vm_viol = get_acpf_data(md, key='max_vm_viol')

    return max_vm_viol


def thermal_viol_max(md):
    '''
    Returns the maximum of thermal limit infeasibilites
    Note: returned value is in MVA
    '''

    if not optimal(md):
        return None
    max_thermal_viol = get_acpf_data(md, key='max_thermal_viol')

    return max_thermal_viol


def vm_UB_viol_pct(md):
    '''
    Returns the number of voltage upper bound infeasibilites
    '''

    if not optimal(md):
        return None
    pct_vm_UB_viol = get_acpf_data(md, key='pct_vm_UB_viol')

    return pct_vm_UB_viol


def vm_LB_viol_pct(md):
    '''
    Returns the number of voltage lower bound infeasibilities
    '''

    if not optimal(md):
        return None
    pct_vm_LB_viol = get_acpf_data(md, key='pct_vm_LB_viol')

    return pct_vm_LB_viol


def vm_viol_pct(md):
    '''
    Returns the number of all voltage infeasibilities
    '''

    if not optimal(md):
        return None
    pct_vm_viol = get_acpf_data(md, key='pct_vm_viol')

    return pct_vm_viol


def thermal_viol_pct(md):
    '''
    Returns the number of thermal limit infeasibilites
    '''

    if not optimal(md):
        return None
    pct_thermal_viol = get_acpf_data(md, key='pct_thermal_viol')

    return pct_thermal_viol


def pf_error_1_norm(md):
    '''
    Returns the 1-norm of real power flow error
    '''

    if not optimal(md):
        return None
    pf_error_1_norm = get_acpf_data(md, key='pf_error_1_norm')

    return pf_error_1_norm


def qf_error_1_norm(md):
    '''
    Returns the 1-norm of reactive power flow error
    '''

    if not optimal(md):
        return None
    qf_error_1_norm = get_acpf_data(md, key='qf_error_1_norm')

    return qf_error_1_norm


def pf_error_inf_norm(md):
    '''
    Returns the infinity-norm of real power flow error
    '''

    if not optimal(md):
        return None
    pf_error_inf_norm = get_acpf_data(md, key='pf_error_inf_norm')

    return pf_error_inf_norm


def qf_error_inf_norm(md):
    '''
    Returns the infinity-norm of reactive power flow error
    '''

    if not optimal(md):
        return None
    qf_error_inf_norm = get_acpf_data(md, key='qf_error_inf_norm')

    return qf_error_inf_norm


def thermal_and_vm_viol_pct(md):

    p1 = thermal_viol_pct(md)
    p2 = vm_viol_pct(md)

    if p1 is None or p2 is not None:
        return None
    val = p1+p2

    return val

def acpf_slack(md):
    '''
    Returns the change in the slack bus real power dispatch in the ACPF solution in MW
    '''

    if not optimal(md):
        return None
    acpf_slack = get_acpf_data(md, key='acpf_slack')

    return acpf_slack

def balance_slack(md):
    '''
    Returns the change in the slack bus real power dispatch in the ACPF solution in MW
    '''

    if not optimal(md):
        return None
    balance_slack = get_acpf_data(md, key='balance_slack')

    return balance_slack

def thermal_viol(md):

    if not optimal(md):
        return None
    th_viol = get_acpf_data(md, key='thermal_viol')

    return th_viol

def vm_viol(md):

    if not optimal(md):
        return None
    vm_viol = get_acpf_data(md, key='vm_viol')

    return vm_viol

def pf_error(md):

    if not optimal(md):
        return None
    # including overwrite_existing since this is the first ACPF data to be collected, so
    # this call will automatically update the acpf data.
    pf_error = get_acpf_data(md, key='pf_error')

    return pf_error

def qf_error(md):

    if not optimal(md):
        return None
    qf_error = get_acpf_data(md, key='qf_error')

    return qf_error
