#  ___________________________________________________________________________
#
#  EGRET: Electrical Grid Research and Engineering Tools
#  Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

'''
lopf tester
'''

import shutil, glob, gc
import sys, getopt
import logging
import copy
import egret.data.test_utils as tu
import egret.data.summary_plot_utils as spu
from pyomo.opt import SolverFactory, TerminationCondition
from egret.common.log import logger
from egret.models.acopf import solve_acopf, create_psv_acopf_model
from egret.models.lopf_fdf import solve_lopf, create_p_lopf_model, create_dense_lopf_model, create_sparse_lopf_model, create_compact_lopf_model
from egret.models.dcopf_losses import solve_dcopf_losses, create_btheta_losses_dcopf_model, create_ptdf_losses_dcopf_model
from egret.models.dcopf import solve_dcopf, create_btheta_dcopf_model, create_ptdf_dcopf_model
from egret.data.lopf_utils import *
#from egret.data.data_utils_deprecated import destroy_dicts_of_fdf, create_dicts_of_ptdf, create_dicts_of_fdf
from egret.parsers.matpower_parser import create_ModelData
#from egret.model_library.transmission.tx_calc import reduce_branches
from os.path import join

current_dir = os.path.dirname(os.path.abspath(__file__))




def generate_test_model_dict(test_model_list):

    test_model_dict = {}
    _kwargs = {'return_model' :False, 'return_results' : False, 'solver_tee' : False}
    tol_keys = ['rel_ptdf_tol', 'rel_qtdf_tol', 'rel_pldf_tol', 'rel_qldf_tol', 'rel_vdf_tol']


    for tm in test_model_list:
        # create empty settings dictionary for each model type
        tmd = dict()
        tmd['kwargs'] = copy.deepcopy(_kwargs)

        # Build ptdf_options based on model name
        _ptdf_options = dict()
        if 'lazy' in tm:
            _ptdf_options['lazy'] = True
            if 'dlopf' in tm or 'clopf' in tm:
                _ptdf_options['lazy_voltage'] = True
        tol = None
        if 'e5' in tm:
            tol = 1e-5
        elif 'e4' in tm:
            tol = 1e-4
        elif 'e3' in tm:
            tol = 1e-3
        elif 'e2' in tm:
            tol = 1e-2
        if any(e in tm for e in ['e5','e4','e3','e2']):
            for k in tol_keys:
                _ptdf_options[k] = tol

        if 'acopf' in tm:
            tmd['solve_func'] = solve_acopf
            tmd['initial_solution'] = 'flat'
            tmd['solver'] = 'ipopt'

        elif 'slopf' in tm:
            tmd['solve_func'] = solve_lccm
            tmd['initial_solution'] = 'basepoint'
            tmd['solver'] = 'gurobi_persistent'
            tmd['kwargs']['include_pf_feasibility_slack'] = False
            tmd['kwargs']['include_qf_feasibility_slack'] = False
            tmd['kwargs']['include_v_feasibility_slack'] = False

        elif 'dlopf' in tm:
            tmd['solve_func'] = solve_fdf
            tmd['initial_solution'] = 'basepoint'
            tmd['solver'] = 'gurobi_persistent'
            tmd['kwargs']['ptdf_options'] = dict(_ptdf_options)
            tmd['kwargs']['include_pf_feasibility_slack'] = False
            tmd['kwargs']['include_qf_feasibility_slack'] = False
            tmd['kwargs']['include_v_feasibility_slack'] = False

        elif 'clopf' in tm:
            tmd['solve_func'] = solve_fdf_simplified
            tmd['initial_solution'] = 'basepoint'
            tmd['solver'] = 'gurobi_persistent'
            tmd['kwargs']['ptdf_options'] = dict(_ptdf_options)
            tmd['kwargs']['include_pf_feasibility_slack'] = False
            tmd['kwargs']['include_qf_feasibility_slack'] = False
            tmd['kwargs']['include_v_feasibility_slack'] = False

        elif 'plopf' in tm:
            tmd['solve_func'] = solve_dcopf_losses
            tmd['initial_solution'] = 'basepoint'
            tmd['solver'] = 'gurobi_persistent'
            tmd['kwargs']['ptdf_options'] = dict(_ptdf_options)
            tmd['kwargs']['dcopf_losses_model_generator'] = create_ptdf_losses_dcopf_model
            tmd['kwargs']['include_pf_feasibility_slack'] = False

        elif 'ptdf' in tm:
            tmd['solve_func'] = solve_dcopf
            tmd['initial_solution'] = 'lossy'
            tmd['solver'] = 'gurobi_persistent'
            tmd['kwargs']['ptdf_options'] = dict(_ptdf_options)
            tmd['kwargs']['dcopf_model_generator'] = create_ptdf_dcopf_model
            tmd['kwargs']['include_pf_feasibility_slack'] = False

        elif 'btheta' in tm:
            if 'qcp' in tm:
                tmd['solve_func'] = solve_dcopf_losses
                tmd['intial_solution'] = 'flat'
                tmd['solver'] = 'gurobi_persistent'
                tmd['kwargs']['dcopf_losses_model_generator'] = create_btheta_losses_dcopf_model
            else:
                tmd['solve_func'] = solve_dcopf
                tmd['initial_solution'] = 'lossy'
                tmd['solver'] = 'gurobi_persistent'
                tmd['kwargs']['dcopf_model_generator'] = create_btheta_dcopf_model

            tmd['kwargs']['include_pf_feasibility_slack'] = False

        # settings to suppress non-lazy D-LOPF and C-LOPF models in large (>1,000 bus) cases
        dense_models = ['dlopf', 'clopf', 'plopf', 'ptdf']
        if 'dlopf' in tm:
            if 'lazy' in tm:
                tmd['suppress_large_cases'] = False
            else:
                tmd['suppress_large_cases'] = False
        if 'clopf' in tm:
            if 'lazy' in tm:
                tmd['suppress_large_cases'] = False
            else:
                tmd['suppress_large_cases'] = False
        else:
            tmd['suppress_large_cases'] = False

        test_model_dict[tm] = copy.deepcopy(tmd)

    return test_model_dict


def get_case_names(flag=None):

    if flag=='misc':
        remove_list = list()
        for key in ['ieee','k','rte','sdet','tamu','pegase']:
            N = len(key)
            remove_list += [c for c in case_names if key in c[-N:]]
        case_list = [c for c in case_names if c not in remove_list]

    elif flag is not None:
        N = len(flag)
        case_list = [c for c in case_names if flag in c[-N:]]

    else:
        case_list = case_names

    return case_list

def get_case_dict():

    # get case dicts
    case_sets = ['ieee','k','rte','sdet','tamu','pegase','misc']
    case_dict = {}
    for k in case_sets:
        case_dict[k] = get_case_names(k)

    return case_dict

def get_test_model_list():
    test_model_list = [
        'acopf',
        'slopf',
        'dlopf_full',
        'dlopf_e4',
        'dlopf_e2',
        'dlopf_lazy_full',
        'dlopf_lazy_e4',
        'dlopf_lazy_e2',
        'clopf_full',
        'clopf_e4',
        'clopf_e2',
        'clopf_lazy_full',
        'clopf_lazy_e4',
        'clopf_lazy_e2',
        'plopf_full',
        'plopf_e4',
        'plopf_e2',
        'plopf_lazy_full',
        'plopf_lazy_e4',
        'plopf_lazy_e2',
        'ptdf_full',
        'ptdf_e4',
        'ptdf_e2',
        'ptdf_lazy_full',
        'ptdf_lazy_e4',
        'ptdf_lazy_e2',
        'btheta',
        # 'btheta_qcp',
    ]
    return test_model_list

def set_acopf_basepoint_min_max(model_data, init_min=0.9, init_max=1.1, **kwargs):
    """
    returns AC basepoint solution and feasible min/max range
     - new min/max range b/c test case may not be feasible in [init_min to init_max]
    """
    md = model_data.clone_in_service()

    acopf_model = create_psv_acopf_model

    md_basept, m, results = solve_acopf(md, "ipopt", acopf_model_generator=acopf_model, return_model=True,
                                        return_results=True, solver_tee=False)

    # exit if base point does not return optimal
    if not results.solver.termination_condition == TerminationCondition.optimal:
        raise Exception('Base case acopf did not return optimal solution')

    # find feasible min and max demand multipliers
    else:
        mult_min = multiplier_loop(md, init=init_min, steps=10, acopf_model=acopf_model)
        mult_max = multiplier_loop(md, init=init_max, steps=10, acopf_model=acopf_model)

    return md_basept, mult_min, mult_max


def multiplier_loop(md, init=0.9, steps=10, acopf_model=create_psv_acopf_model):
    '''
    init < 1 searches for the lowest demand multiplier >= init that has an optimal acopf solution
    init > 1 searches for the highest demand multiplier <= init that has an optimal acopf solution
    steps determines the increments in [init, 1] where the search is made
    '''

    loads = dict(md.elements(element_type='load'))

    # step size
    inc = abs(1 - init) / steps

    # initial loads
    init_p_loads = {k: loads[k]['p_load'] for k in loads.keys()}
    init_q_loads = {k: loads[k]['q_load'] for k in loads.keys()}

    # loop
    final_mult = None
    for step in range(0, steps):

        # for finding minimum
        if init < 1:
            mult = round(init - step * inc, 4)

        # for finding maximum
        elif init > 1:
            mult = round(init - step * inc, 4)

        # adjust load from init_min
        for k in loads.keys():
            loads[k]['p_load'] = init_p_loads[k] * mult
            loads[k]['q_load'] = init_q_loads[k] * mult

        try:
            md_, results = solve_acopf(md, "ipopt", acopf_model_generator=acopf_model, return_model=False,
                                       return_results=True, solver_tee=False)

            for k in loads.keys(): # revert back to initial loadings
                loads[k]['p_load'] = init_p_loads[k]
                loads[k]['q_load'] = init_q_loads[k]

            final_mult = mult
            print('mult={} has an acceptable solution.'.format(mult))
            break

        except Exception:
            print('mult={} raises an error. Continuing search.'.format(mult))

    if final_mult is None:
        print('Found no acceptable solutions with mult != 1. Try init between 1 and {}.'.format(mult))
        final_mult = 1

    return final_mult


def create_new_model_data(model_data, mult, loss_adj=1.0):
    md = model_data.clone_in_service()

    loads = dict(md.elements(element_type='load'))

    # initial loads
    init_p_loads = {k: loads[k]['p_load'] for k in loads.keys()}
    init_q_loads = {k: loads[k]['q_load'] for k in loads.keys()}

    # multiply loads
    for k in loads.keys():
        loads[k]['p_load'] = init_p_loads[k] * mult * loss_adj
        loads[k]['q_load'] = init_q_loads[k] * mult * loss_adj

    md.data['system']['mult'] = mult

    return md


def inner_loop_solves(md_basepoint, md_flat, md_lossy, test_model_list):
    '''
    solve models in test_model_dict (ideally, only one model is passed here)
    loads are multiplied by mult
    sensitivities from md_basepoint or md_flat as appropriate for the model being solved
    '''

    bus_attrs = md_flat.attributes(element_type='bus')
    num_bus = len(bus_attrs['names'])
    mult = md_flat.data['system']['mult']

    # will try to relax constraints if initial solves are infeasible
    relaxations = [None, 'include_v_feasibility_slack', 'include_qf_feasibility_slack', 'include_pf_feasibility_slack']

    for tm in test_model_list:
        tm_dict = generate_test_model_dict([tm])[tm]
        if tm_dict['suppress_large_cases'] and num_bus > 1000:
            continue
        print('>>>>> BEGIN SOLVE: {} / {} <<<<<'.format(tm,mult))
        solve_func = tm_dict['solve_func']
        initial_solution = tm_dict['initial_solution']
        solver = tm_dict['solver']
        kwargs = tm_dict['kwargs']
        if initial_solution == 'flat':
            md_input = md_flat
        elif initial_solution == 'basepoint':
            md_input = md_basepoint
        elif initial_solution == 'lossy':
            md_input = md_lossy
        else:
            raise Exception('test_model_dict must provide valid initial_solution')

        # Apply progressive relaxations if initial solve is infeasible
        for r in relaxations:
            if r is not None:
                reactive_relax = any(s in r for s in ['_qf_', '_v_'])
                reactive_model = any(m in tm for m in ['slopf', 'dlopf', 'clopf'])
                if reactive_relax and not reactive_model:
                    continue
                kwargs[r] = True
                logger.critical('...applying relaxation with {}'.format(r))
            try:
                md_out = solve_func(md_input, solver=solver, **kwargs)
                logger.critical('\t COST = ${:,.2f}'.format(md_out.data['system']['total_cost']))
                logger.critical('\t TIME = {:.5f} seconds'.format(md_out.data['results']['time']))
                is_feasible = True
            except Exception as e:
                is_feasible = False
                model_error = str(e)
                logger.critical('failed: {}'.format(model_error))
            # end loop if solve is successful
            if is_feasible:
                break

        # create results object if all relaxations failed
        if not is_feasible:
            md_out = md_input
            md_out.data['results'] = {}
            md_out.data['results']['termination'] = 'infeasible'
            md_out.data['results']['exception'] = model_error

        # return relaxation kwargs to False
        for r in relaxations:
            kwargs[r] = False

        record_results(tm, md_out)


def record_results(idx, md):
    '''
    writes model data (md) object to .json file
    '''

    #destroy_dicts_of_fdf(md)

    mult = md.data['system']['mult']
    filename = md.data['system']['model_name'] + '_' + idx + '_{0:04.0f}'.format(mult * 1000)
    md.data['system']['filename'] = filename
    md.write_to_json(filename)
    print('...out: {}.json'.format(filename))

    tu.repopulate_acpf_to_modeldata(md)
    md.write_to_json(filename)

    if md.data['results']['termination'] == 'optimal':
        del md
        gc.collect()
    else:
        del md.data['results']
        del md.data['system']['filename']
        gc.collect()


def create_testcase_directory(test_case):
    # directory locations
    cwd = os.getcwd()
    case_folder, case = os.path.split(test_case)
    case, ext = os.path.splitext(case)
    current_dir, current_file = os.path.split(os.path.realpath(__file__))

    # move to case directory
    source = os.path.join(cwd, case + '_*.json')
    destination = get_solution_file_location(test_case)

    if not os.path.exists(destination):
        os.makedirs(destination)

    if not glob.glob(source):
        print('No files to move.')
    else:
        print('dest: {}'.format(destination))

        for src in glob.glob(source):
            print('src:  {}'.format(src))
            folder, file = os.path.split(src)
            dest = os.path.join(destination, file)  # full destination path will overwrite existing files
            shutil.move(src, dest)

    return destination


def calc_loss_adj(model_data):
    # Calculate demand multiplier to adjust for losses
    from egret.model_library.transmission.tx_utils import dict_of_bus_fixed_shunts

    buses = dict(model_data.elements(element_type='bus'))
    loads = dict(model_data.elements(element_type='load'))
    gens = dict(model_data.elements(element_type='generator'))
    shunts = dict(model_data.elements(element_type='shunt'))
    bus_bs, bus_gs = dict_of_bus_fixed_shunts(buses, shunts)
    tot_d = sum(loads[k]['p_load'] for k in loads.keys()) + sum(bus_gs[k] * buses[k]['vm']**2 for k in buses.keys())
    tot_g = sum(gens[k]['pg'] for k in gens.keys())
    loss_adj = tot_g / tot_d

    return loss_adj

def solve_approximation_models(test_case, test_model_list, init_min=0.9, init_max=1.1, steps=20):
    '''
    1. initialize base case and demand range
    2. loop over demand values
    3. record results to .json files
    '''

    _md_flat = create_ModelData(test_case)
    _len_bus = tu.num_buses(_md_flat)
    _len_branch = tu.num_branches(_md_flat)
    _len_cycle = _len_branch - _len_bus + 1
    tml = test_model_list

    logger.critical("Beginning solution loop for: {}".format(_md_flat.data['system']['model_name']))

    _md_basept, min_mult, max_mult = set_acopf_basepoint_min_max(_md_flat, init_min, init_max)
    if 'acopf' not in tml:
        tml.append('acopf')

    # Calculate sensitivity multiplers, and make sure the base case mult=1 is included
    inc = (max_mult - min_mult) / steps
    multipliers = [round(min_mult + step * inc, 4) for step in range(0, steps + 1)]
    if 1.0 not in multipliers:
        multipliers.append(1.0)
        multipliers.sort()

    loss_adj = calc_loss_adj(_md_basept)
    branches = dict(_md_basept.elements(element_type='branch'))
    #active_branches = reduce_branches(branches, _len_cycle)
    incl_lopf = any('lopf' in model for model in tml)
    incl_loss = any('ptdf' in model for model in tml) or any('btheta' in model for model in tml)
    if incl_lopf:
        pass
        #create_dicts_of_fdf(_md_basept)
    else:
        del _md_basept
    if incl_loss:
        pass
        #create_dicts_of_ptdf(_md_flat, active_branches=active_branches)

    for mult in multipliers:
        md_flat = create_new_model_data(_md_flat, mult)
        if incl_lopf:
            md_basept = create_new_model_data(_md_basept, mult)
        else:
            md_basept = None
        if incl_loss:
            md_lossy = create_new_model_data(_md_flat, mult, loss_adj=loss_adj)
        else:
            md_lossy = None

        inner_loop_solves(md_basept, md_flat, md_lossy, test_model_list)

    create_testcase_directory(test_case)



def run_test_loop(idx=None, tml=None, show_plot=False, log_level=logging.CRITICAL):
    """
    solves models and generates plots for test case at test_cases[idx] or a default case
    """

    logger = logging.getLogger('egret')
    logger.setLevel(log_level)

    # Select default case
    if idx is None:
#        test_case = join('../../../download/pglib-opf-master/', 'pglib_opf_case3_lmbd.m')
        test_case = join('../../../download/pglib-opf-master/', 'pglib_opf_case5_pjm.m')
#        test_case = join('../../download/pglib-opf-master/', 'pglib_opf_case30_ieee.m')
#        test_case = join('../../download/pglib-opf-master/', 'pglib_opf_case24_ieee_rts.m')
#        test_case = join('../../../download/pglib-opf-master/', 'pglib_opf_case118_ieee.m')
#        test_case = join('../../download/pglib-opf-master/', 'pglib_opf_case300_ieee.m')
    else:
        test_case=idx_to_test_case(idx)

    # file path correction if running from /models/
    if os.path.basename(os.getcwd()) == 'models':
        test_case = test_case[3:]

    # Select test model list
    if tml is None:
        tml = get_test_model_list()

    ## Model solves
    solve_approximation_models(test_case, tml, init_min=0.95, init_max=1.05, steps=10)

    ## Generate summary data
    #spu.create_full_summary(test_case, tml, show_plot=show_plot)


def run_nominal_test(idx=None, tml=None, show_plot=False, log_level=logging.CRITICAL):
    """
    solves models and generates plots for test case at test_cases[idx] or a default case
    """

    logger = logging.getLogger('egret')
    logger.setLevel(log_level)

    # Select default case
    if idx is None:
#        test_case = join('../../../download/pglib-opf-master/', 'pglib_opf_case3_lmbd.m')
        test_case = join('../../../download/pglib-opf-master/', 'pglib_opf_case5_pjm.m')
#        test_case = join('../../download/pglib-opf-master/', 'pglib_opf_case30_ieee.m')
#        test_case = join('../../download/pglib-opf-master/', 'pglib_opf_case24_ieee_rts.m')
#        test_case = join('../../../download/pglib-opf-master/', 'pglib_opf_case118_ieee.m')
#        test_case = join('../../download/pglib-opf-master/', 'pglib_opf_case300_ieee.m')
    else:
        test_case=idx_to_test_case(idx)

    # file path correction if running from /models/
    if os.path.basename(os.getcwd()) == 'models':
        test_case = test_case[3:]

    # Select test model list
    if tml is None:
        tml = get_test_model_list()

    ## Model solves
    md_flat = create_ModelData(test_case)
    _len_bus = tu.num_buses(md_flat)
    _len_branch = tu.num_branches(md_flat)
    _len_cycle = _len_branch - _len_bus + 1

    print('>>>>> BEGIN SOLVE: acopf <<<<<')
    md_basept = solve_acopf(md_flat,solver='ipopt',solver_tee=False)
    logger.critical('\t COST = ${:,.2f}'.format(md_basept.data['system']['total_cost']))
    logger.critical('\t TIME = {:.5f} seconds'.format(md_basept.data['results']['time']))
    md_basept.data['system']['mult'] = 1
    record_results('acopf', md_basept)

    # remove acopf from further solves since it is already solved above
    if 'acopf' in tml:
        tml.remove('acopf')

    ## put the sensitivities into modeData so they don't need to be recalculated for each model
    loss_adj = calc_loss_adj(md_basept)
    branches = dict(md_basept.elements(element_type='branch'))
    #active_branches = reduce_branches(branches, _len_cycle)
    md_flat = create_new_model_data(md_flat, 1.0)
    if any('lopf' in model for model in tml):
        #create_dicts_of_fdf(md_basept)
        md_basept = create_new_model_data(md_basept, 1.0)
    else:
        md_basept = None
    if any('ptdf' in model for model in tml) or any('btheta' in model for model in tml):
        #create_dicts_of_ptdf(md_flat, active_branches=active_branches)
        md_lossy = create_new_model_data(md_flat, 1.0, loss_adj=loss_adj)
    else:
        md_lossy = None

    inner_loop_solves(md_basept, md_flat, md_lossy, tml)

    create_testcase_directory(test_case)

    #summarize_nominal_data(test_case=test_case, show_plot=show_plot)

def summarize_nominal_data(idx=0, tml=None, test_case=None,show_plot=True, log_level=None):

    if test_case is None:
        test_case = idx_to_test_case(idx)

    if tml is None:
        tml = get_test_model_list()

    #spu.update_data_file(test_case)
    #spu.update_data_tables()
    #spu.acpf_violations_plot(test_case, tml, show_plot=show_plot)
    spu.create_full_summary(test_case, tml, show_plot=show_plot)


def batch(arg, subbatch=run_test_loop):

    idxA0 = 0
    #idxA0 = case_names.index('pglib_opf_case89_pegase')  ## redefine first case of A
    idxA = case_names.index('pglib_opf_case1354_pegase')  ## < 1000 buses
    idxB = case_names.index('pglib_opf_case2736sp_k')  ## 1354 - 2383 buses
    idxC = case_names.index('pglib_opf_case6468_rte')  ## 2383 - 4661 buses
    idxD = case_names.index('pglib_opf_case13659_pegase')  ## 6468 - 10000 buses
    idxE = case_names.index('pglib_opf_case13659_pegase') + 1  ## 13659 buses
    idx7000 = case_names.index('pglib_opf_case9241_pegase')

    if arg == 'A':
        idx_list = list(range(idxA0,idxA))
    elif arg == 'B':
        idx_list = list(range(idxA,idxB))
    elif arg == 'C':
        idx_list = list(range(idxB,idxC))
    elif arg == 'D':
        idx_list = list(range(idxC,idxD))
    elif arg == 'E':
        idx_list = list(range(idxD,idxE))
    elif arg == 'X':
        idx_list = list(range(0,idxE))      # all cases
    elif arg == 'Y':
        idx_list = list(range(idxA,idxE))   # all >1,000 bus cases
    elif arg == 'Z':
        idx_list = list(range(idxA,idx7000))   # all >1,000 bus, <7,000 bus cases

    for idx in idx_list:
        subbatch(idx, show_plot=False, log_level=logging.CRITICAL)


def main(argv):

    message = 'test_approximations.py usage must include a batch or a case: \n'
    message += '  -b --batch=<letter>         ::  run a preset batch of cases \n'
    message += ' \t A \t cases 3-588 \n'
    message += ' \t B \t cases 1354-2383 \n'
    message += ' \t C \t cases 2736-4661 \n'
    message += ' \t D \t cases 6468-10000 \n'
    message += ' \t E \t case 13659 \n'
    message += ' \t X \t all cases \n'
    message += ' \t Y \t all cases >1000 \n'
    message += ' \t Z \t all cases >1000, <7000 \n'
    message += '  -c --case=<case_name/idx>   ::  run a specific case or case index \n'
    message += '  -d --data-only              ::  run summary data only \n'
    message += '  -n --nominal                ::  run a specific case with nominal demand only \n'
    message += '  -q --quick                  ::  same as nominal option \n'

    try:
        opts, args = getopt.getopt(argv, "b:c:dnq", ["batch=", "case=", "data", "nominal", "quick"])
    except getopt.GetoptError:
        print(message)
        sys.exit(2)

    batch_run = False
    quick_run = False
    data_run = False
    case_run = False
    for opt, arg in opts:
        if opt in ('-b', '--batch'):
            batch_run = True
            batch_arg = arg
        elif opt in ('-c', '--case'):
            case_run = True
            case_arg = arg
        elif opt in ('-d', '--data-only'):
            data_run = True
        elif opt in ('-n', '--nominal'):
            quick_run = True
        elif opt in ('-q', '--quick'):
            quick_run = True

    if batch_run:
        if quick_run:
            batch(batch_arg, subbatch=run_nominal_test)
        elif data_run:
            batch(batch_arg, subbatch=summarize_nominal_data)
        else:
            batch(batch_arg, subbatch=run_test_loop)
    elif case_run:
        if quick_run:
            run_nominal_test(case_arg)
        elif data_run:
            summarize_nominal_data(idx=case_arg)
        else:
            run_test_loop(case_arg)
    else:
        print(message)


if __name__ == '__main__':
    main(sys.argv[1:])
