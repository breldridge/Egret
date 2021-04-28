#  ___________________________________________________________________________
#
#  EGRET: Electrical Grid Research and Engineering Tools
#  Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
This module provides functions that create the modules for typical ACPF formulations.

#TODO: document this with examples
"""
import pyomo.environ as pe
import egret.model_library.transmission.tx_utils as tx_utils
import egret.model_library.transmission.tx_calc as tx_calc
import egret.model_library.transmission.bus as libbus
import egret.model_library.transmission.branch as libbranch
import egret.model_library.transmission.gen as libgen
from egret.data.data_utils import zip_items

from egret.model_library.defn import CoordinateType
from math import pi
from collections import OrderedDict


def _include_feasibility_slack(model, bus_attrs, gen_attrs, bus_p_loads, bus_q_loads, penalty=9999, quadratic_penalty=False):
    import egret.model_library.decl as decl
    slack_init = {k: 0 for k in bus_attrs['names']}

    #slack_bounds = {k: (0, sum(bus_p_loads.values())) for k in bus_attrs['names']}
    #slack_bounds = {k: (0, None) for k in bus_attrs['names']}
    slack_bounds = {k: (0, 0.01) for k in bus_attrs['names']}
    decl.declare_var('p_slack_pos', model=model, index_set=bus_attrs['names'],
                     initialize=slack_init, bounds=slack_bounds
                     )
    decl.declare_var('p_slack_neg', model=model, index_set=bus_attrs['names'],
                     initialize=slack_init, bounds=slack_bounds
                     )
    slack_bounds = {k: (0, sum(bus_q_loads.values())) for k in bus_attrs['names']}
    decl.declare_var('q_slack_pos', model=model, index_set=bus_attrs['names'],
                     initialize=slack_init, bounds=slack_bounds
                     )
    decl.declare_var('q_slack_neg', model=model, index_set=bus_attrs['names'],
                     initialize=slack_init, bounds=slack_bounds
                     )
    p_rhs_kwargs = {'include_feasibility_slack_pos':'p_slack_pos','include_feasibility_slack_neg':'p_slack_neg'}
    q_rhs_kwargs = {'include_feasibility_slack_pos':'q_slack_pos','include_feasibility_slack_neg':'q_slack_neg'}
    #p_penalty = penalty * (max([gen_attrs['p_cost'][k]['values'][1] for k in gen_attrs['names']]) + 1)
    #q_penalty = penalty * (max(gen_attrs.get('q_cost', gen_attrs['p_cost'])[k]['values'][1] for k in gen_attrs['names']) + 1)
    p_penalty = penalty
    q_penalty = penalty
    penalty_expr = sum(p_penalty * (model.p_slack_pos[bus_name] + model.p_slack_neg[bus_name])
                    + q_penalty * (model.q_slack_pos[bus_name] + model.q_slack_neg[bus_name])
                    for bus_name in bus_attrs['names'])

    if quadratic_penalty:
        penalty_expr += penalty * sum( model.p_slack_pos[bus_name]**2 + model.p_slack_neg[bus_name]**2
                    + model.q_slack_pos[bus_name]**2 + model.q_slack_neg[bus_name]**2
                    for bus_name in bus_attrs['names'])

    return p_rhs_kwargs, q_rhs_kwargs, penalty_expr


def create_psv_acpf_model(model_data, include_feasibility_slack=False):
    # model_data.return_in_service()
    # md = model_data
    md = model_data.clone_in_service()
    tx_utils.scale_ModelData_to_pu(md, inplace = True)

    gens = dict(md.elements(element_type='generator'))
    buses = dict(md.elements(element_type='bus'))
    branches = dict(md.elements(element_type='branch'))
    loads = dict(md.elements(element_type='load'))
    shunts = dict(md.elements(element_type='shunt'))

    gen_attrs = md.attributes(element_type='generator')
    bus_attrs = md.attributes(element_type='bus')
    branch_attrs = md.attributes(element_type='branch')
    load_attrs = md.attributes(element_type='load')
    shunt_attrs = md.attributes(element_type='shunt')

    inlet_branches_by_bus, outlet_branches_by_bus = \
        tx_utils.inlet_outlet_branches_by_bus(branches, buses)
    gens_by_bus = tx_utils.gens_by_bus(buses, gens)
    buses_with_gens = tx_utils.buses_with_gens(gens)

    model = pe.ConcreteModel()

    ### declare (and fix) the loads at the buses
    bus_p_loads, bus_q_loads = tx_utils.dict_of_bus_loads(buses, loads)

    libbus.declare_var_pl(model, bus_attrs['names'], initialize=bus_p_loads)
    libbus.declare_var_ql(model, bus_attrs['names'], initialize=bus_q_loads)
    model.pl.fix()
    model.ql.fix()

    ### declare the fixed shunts at the buses
    bus_bs_fixed_shunts, bus_gs_fixed_shunts = tx_utils.dict_of_bus_fixed_shunts(buses, shunts)

    ### declare the polar voltages
    libbus.declare_var_vm(model, bus_attrs['names'], initialize=bus_attrs['vm'])

    va_bounds = {k: (-pi, pi) for k in bus_attrs['va']}
    libbus.declare_var_va(model, bus_attrs['names'], initialize=bus_attrs['va'])

    ### declare the generator real and reactive power
    libgen.declare_var_pg(model, gen_attrs['names'], initialize=gen_attrs['pg'])

    libgen.declare_var_qg(model, gen_attrs['names'], initialize=gen_attrs['qg'])

    ### include the feasibility slack for the bus balances
    p_rhs_kwargs = {}
    q_rhs_kwargs = {}
    if include_feasibility_slack:
        print('...relaxing power balance...')
        p_rhs_kwargs, q_rhs_kwargs, penalty_expr = _include_feasibility_slack(model, bus_attrs, gen_attrs, bus_p_loads, bus_q_loads, quadratic_penalty=True)

    ### In a system with N buses and G generators, there are then 2(N-1)-(G-1) unknowns.
    ### fix the reference bus
    ref_bus = md.data['system']['reference_bus']
    model.vm[ref_bus].fixed = True
    model.va[ref_bus].fixed = True

    for bus_name in bus_attrs['names']:
        if bus_name != ref_bus and bus_name in buses_with_gens:
            model.vm[bus_name].fixed = True
            for gen_name in gens_by_bus[bus_name]:
                model.pg[gen_name].fixed = True

    ### declare the current flows in the branches
    vr_init = {k: bus_attrs['vm'][k] * pe.cos(bus_attrs['va'][k]) for k in bus_attrs['vm']}
    vj_init = {k: bus_attrs['vm'][k] * pe.sin(bus_attrs['va'][k]) for k in bus_attrs['vm']}
    s_max = {k: branches[k]['rating_long_term'] for k in branches.keys()}
    s_lbub = dict()
    for k in branches.keys():
        if s_max[k] is None:
            s_lbub[k] = (None, None)
        else:
            s_lbub[k] = (-s_max[k],s_max[k])
    pf_init = dict()
    pt_init = dict()
    qf_init = dict()
    qt_init = dict()
    for branch_name, branch in branches.items():
        from_bus = branch['from_bus']
        to_bus = branch['to_bus']
        y_matrix = tx_calc.calculate_y_matrix_from_branch(branch)
        ifr_init = tx_calc.calculate_ifr(vr_init[from_bus], vj_init[from_bus], vr_init[to_bus],
                                         vj_init[to_bus], y_matrix)
        ifj_init = tx_calc.calculate_ifj(vr_init[from_bus], vj_init[from_bus], vr_init[to_bus],
                                         vj_init[to_bus], y_matrix)
        itr_init = tx_calc.calculate_itr(vr_init[from_bus], vj_init[from_bus], vr_init[to_bus],
                                         vj_init[to_bus], y_matrix)
        itj_init = tx_calc.calculate_itj(vr_init[from_bus], vj_init[from_bus], vr_init[to_bus],
                                         vj_init[to_bus], y_matrix)
        pf_init[branch_name] = tx_calc.calculate_p(ifr_init, ifj_init, vr_init[from_bus], vj_init[from_bus])
        pt_init[branch_name] = tx_calc.calculate_p(itr_init, itj_init, vr_init[to_bus], vj_init[to_bus])
        qf_init[branch_name] = tx_calc.calculate_q(ifr_init, ifj_init, vr_init[from_bus], vj_init[from_bus])
        qt_init[branch_name] = tx_calc.calculate_q(itr_init, itj_init, vr_init[to_bus], vj_init[to_bus])

    libbranch.declare_var_pf(model=model,
                             index_set=branch_attrs['names'],
                             initialize=pf_init
                             )

    libbranch.declare_var_pt(model=model,
                             index_set=branch_attrs['names'],
                             initialize=pt_init
                             )

    libbranch.declare_var_qf(model=model,
                             index_set=branch_attrs['names'],
                             initialize=qf_init
                             )

    libbranch.declare_var_qt(model=model,
                             index_set=branch_attrs['names'],
                             initialize=qt_init
                             )

    ### declare the branch power flow constraints
    libbranch.declare_eq_branch_power(model=model,
                                      index_set=branch_attrs['names'],
                                      branches=branches,
                                      branch_attrs=branch_attrs,
                                      coordinate_type=CoordinateType.POLAR
                                      )
    ### declare the pq balances
    libbus.declare_eq_p_balance(model=model,
                                index_set=bus_attrs['names'],
                                bus_p_loads=bus_p_loads,
                                gens_by_bus=gens_by_bus,
                                bus_gs_fixed_shunts=bus_gs_fixed_shunts,
                                inlet_branches_by_bus=inlet_branches_by_bus,
                                outlet_branches_by_bus=outlet_branches_by_bus,
                                coordinate_type=CoordinateType.POLAR,
                                **p_rhs_kwargs
                                )

    libbus.declare_eq_q_balance(model=model,
                                index_set=bus_attrs['names'],
                                bus_q_loads=bus_q_loads,
                                gens_by_bus=gens_by_bus,
                                bus_bs_fixed_shunts=bus_bs_fixed_shunts,
                                inlet_branches_by_bus=inlet_branches_by_bus,
                                outlet_branches_by_bus=outlet_branches_by_bus,
                                coordinate_type=CoordinateType.POLAR,
                                **q_rhs_kwargs
                                )

    obj_expr = 0.0
    if include_feasibility_slack:
        obj_expr += penalty_expr

    model.obj = pe.Objective(expr=obj_expr)

    from pyomo.contrib.preprocessing.plugins.init_vars import InitMidpoint
    InitMidpoint().apply_to(model)

    return model, md



def solve_acpf(model_data,
                solver,
                timelimit = None,
                solver_tee = True,
                symbolic_solver_labels = False,
                options = None,
                acpf_model_generator = create_psv_acpf_model,
                return_model = False,
                return_results = False,
                **kwargs):
    '''
    Create and solve a new acpf model

    Parameters
    ----------
    model_data : egret.data.ModelData
        An egret ModelData object with the appropriate data loaded.
    solver : str or pyomo.opt.base.solvers.OptSolver
        Either a string specifying a pyomo solver name, or an instantiated pyomo solver
    timelimit : float (optional)
        Time limit for dcopf run. Default of None results in no time
        limit being set.
    solver_tee : bool (optional)
        Display solver log. Default is True.
    symbolic_solver_labels : bool (optional)
        Use symbolic solver labels. Useful for debugging; default is False.
    options : dict (optional)
        Other options to pass into the solver. Default is dict().
    acpf_model_generator : function (optional)
        Function for generating the acpf model. Default is
        egret.models.acpf.create_psv_acpf_model
    return_model : bool (optional)
        If True, returns the pyomo model object
    return_results : bool (optional)
        If True, returns the pyomo results object
    kwargs : dictionary (optional)
        Additional arguments for building model
    '''

    import pyomo.environ as pe
    from pyomo.environ import value
    from egret.common.solver_interface import _solve_model
    from egret.model_library.transmission.tx_utils import \
        scale_ModelData_to_pu, unscale_ModelData_to_pu

    m, md = acpf_model_generator(model_data, **kwargs)

    m.dual = pe.Suffix(direction=pe.Suffix.IMPORT)

    m, results = _solve_model(m,solver,timelimit=timelimit,solver_tee=solver_tee,
                              symbolic_solver_labels=symbolic_solver_labels,options=options)

    # save results data to ModelData object
    gens = dict(md.elements(element_type='generator'))
    buses = dict(md.elements(element_type='bus'))
    branches = dict(md.elements(element_type='branch'))
    bus_attrs = md.attributes(element_type='bus')

    balance_slack = 0
    if hasattr(m, 'p_slack_pos'):
        balance_slack += sum( value(m.p_slack_pos[bus_name]) for bus_name in bus_attrs['names'])
    if hasattr(m, 'p_slack_neg'):
        balance_slack += sum( value(m.p_slack_neg[bus_name]) for bus_name in bus_attrs['names'])
    if hasattr(m, 'p_slack'):
        balance_slack += sum( abs(value(m.p_slack[bus_name])) for bus_name in bus_attrs['names'])
    if hasattr(m, 'q_slack_pos'):
        balance_slack += sum( value(m.q_slack_pos[bus_name]) for bus_name in bus_attrs['names'])
    if hasattr(m, 'q_slack_neg'):
        balance_slack += sum( value(m.q_slack_neg[bus_name]) for bus_name in bus_attrs['names'])
    if hasattr(m, 'q_slack'):
        balance_slack += sum( abs(value(m.q_slack[bus_name])) for bus_name in bus_attrs['names'])
    md.data['system']['balance_slack'] = balance_slack

    for g,g_dict in gens.items():
        g_dict['pg'] = value(m.pg[g])
        g_dict['qg'] = value(m.qg[g])

    for b,b_dict in buses.items():
        b_dict['lmp'] = value(m.dual[m.eq_p_balance[b]])
        b_dict['qlmp'] = value(m.dual[m.eq_q_balance[b]])
        b_dict['pl'] = value(m.pl[b])
        if hasattr(m, 'vj'):
            b_dict['vm'] = tx_calc.calculate_vm_from_vj_vr(value(m.vj[b]), value(m.vr[b]))
            b_dict['va'] = tx_calc.calculate_va_from_vj_vr(value(m.vj[b]), value(m.vr[b]))
        else:
            b_dict['vm'] = value(m.vm[b])
            b_dict['va'] = value(m.va[b])

    for k, k_dict in branches.items():
        if hasattr(m,'pf'):
            k_dict['pf'] = value(m.pf[k])
            k_dict['pt'] = value(m.pt[k])
            k_dict['qf'] = value(m.qf[k])
            k_dict['qt'] = value(m.qt[k])
        if hasattr(m,'irf'):
            b = k_dict['from_bus']
            k_dict['pf'] = value(tx_calc.calculate_p(value(m.ifr[k]), value(m.ifj[k]), value(m.vr[b]), value(m.vj[b])))
            k_dict['qf'] = value(tx_calc.calculate_q(value(m.ifr[k]), value(m.ifj[k]), value(m.vr[b]), value(m.vj[b])))
            b = k_dict['to_bus']
            k_dict['pt'] = value(tx_calc.calculate_p(value(m.itr[k]), value(m.itj[k]), value(m.vr[b]), value(m.vj[b])))
            k_dict['qt'] = value(tx_calc.calculate_q(value(m.itr[k]), value(m.itj[k]), value(m.vr[b]), value(m.vj[b])))


    unscale_ModelData_to_pu(md, inplace=True)

    if return_model and return_results:
        return md, m, results
    elif return_model:
        return md, m
    elif return_results:
        return md, results
    return md

# if __name__ == '__main__':
#     import os
#     from egret.parsers.matpower_parser import create_ModelData
#
#     path = os.path.dirname(__file__)
#     filename = 'pglib_opf_case3_lmbd.m'
#     matpower_file = os.path.join(path, '../../download/pglib-opf/', filename)
#     md = create_ModelData(matpower_file)
#     kwargs = {'include_feasibility_slack':'True'}
#     md = solve_acpf(md, "ipopt",**kwargs)
#
