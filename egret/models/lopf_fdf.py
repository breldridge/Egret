#  ___________________________________________________________________________
#
#  EGRET: Electrical Grid Research and Engineering Tools
#  Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
This module provides functions that create the modules for typical DCOPF formulations.

#TODO: document this with examples
"""
import logging
import pyomo.environ as pe
import numpy as np
import egret.model_library.transmission.tx_utils as tx_utils
import egret.model_library.transmission.tx_calc as tx_calc
import egret.model_library.transmission.bus as libbus
import egret.model_library.transmission.branch as libbranch
import egret.model_library.transmission.gen as libgen
import egret.model_library.decl as decl
import egret.common.lazy_ptdf_utils as lpu
import egret.data.ptdf_utils as ptdf_utils

from egret.model_library.defn import CoordinateType, ApproximationType, BasePointType
from egret.data.data_utils import map_items, zip_items
from egret.models.copperplate_dispatch import (_include_system_feasibility_slack,
                                               _validate_and_extract_slack_penalty,
                                               create_copperplate_dispatch_approx_model)
from egret.common.log import logger
from math import pi, radians


def _include_pq_feasibility_slack(model, bus_names, bus_p_loads, bus_q_loads, gens_by_bus, gen_attrs,
                                  p_marginal_slack_penalty, q_marginal_slack_penalty):
    import egret.model_library.decl as decl
    slack_init = 0
    p_load_shed_bounds = sum(tx_utils.load_shed_limit(bus_p_loads[k], gens_by_bus[k], gen_attrs['p_min']) for k in bus_names)
    q_load_shed_bounds = sum(tx_utils.load_shed_limit(bus_q_loads[k], gens_by_bus[k], gen_attrs['q_min']) for k in bus_names)
    decl.declare_var('p_load_shed', model=model, index_set=None,
                     initialize=0, bounds=(0,p_load_shed_bounds)
                     )
    decl.declare_var('q_load_shed', model=model, index_set=None,
                     initialize=0, bounds=(0,q_load_shed_bounds)
                     )
    p_over_gen_bounds = sum(tx_utils.over_gen_limit(bus_p_loads[k], gens_by_bus[k], gen_attrs['p_max']) for k in bus_names)
    q_over_gen_bounds = sum(tx_utils.over_gen_limit(bus_q_loads[k], gens_by_bus[k], gen_attrs['q_max']) for k in bus_names)
    decl.declare_var('p_over_generation', model=model, index_set=None,
                     initialize=0, bounds=(0,p_over_gen_bounds)
                     )
    decl.declare_var('q_over_generation', model=model, index_set=None,
                     initialize=0, bounds=(0,q_over_gen_bounds)
                     )
    p_rhs_kwargs = {'include_feasibility_load_shed':'p_load_shed', 'include_feasibility_over_generation':'p_over_generation'}
    q_rhs_kwargs = {'include_reactive_load_shed':'q_load_shed', 'include_reactive_over_generation':'q_over_generation'}

    p_penalty = p_marginal_slack_penalty * (model.p_load_shed + model.p_over_generation)
    q_penalty = q_marginal_slack_penalty * (model.q_load_shed + model.q_over_generation)
    penalty_expr = p_penalty + q_penalty

    return p_rhs_kwargs, q_rhs_kwargs, penalty_expr

def _include_feasibility_slack(model, bus_names, bus_p_loads, gens_by_bus, gen_attrs, p_marginal_slack_penalty):
    import egret.model_library.decl as decl

    load_shed_bounds  = {k: (0, tx_utils.load_shed_limit(bus_p_loads[k], gens_by_bus[k], gen_attrs['p_min'])) for k in bus_names}
    decl.declare_var('p_load_shed', model=model, index_set=bus_names,
                     initialize=0., bounds=load_shed_bounds
                     )
    over_gen_bounds = {k: (0, tx_utils.over_gen_limit(bus_p_loads[k], gens_by_bus[k], gen_attrs['p_max'])) for k in bus_names}
    decl.declare_var('p_over_generation', model=model, index_set=bus_names,
                     initialize=0., bounds=over_gen_bounds
                     )

    p_rhs_kwargs = {'include_feasibility_load_shed':'p_load_shed', 'include_feasibility_over_generation':'p_over_generation'}

    penalty_expr = sum(p_marginal_slack_penalty * (model.p_load_shed[bus_name] + model.p_over_generation[bus_name])
                    for bus_name in bus_names)
    return p_rhs_kwargs, penalty_expr

def create_sparse_lopf_model(model_data, include_angle_diff_limits=False, include_feasibility_slack=False):
    #TODO: change to S-LOPF model formulation
    md = model_data.clone_in_service()
    tx_utils.scale_ModelData_to_pu(md, inplace = True)

    gens = dict(md.elements(element_type='generator'))
    buses = dict(md.elements(element_type='bus'))
    branches = dict(md.elements(element_type='branch'))
    loads = dict(md.elements(element_type='load'))
    shunts = dict(md.elements(element_type='shunt'))

    dc_branches = dict(md.elements(element_type='dc_branch'))

    gen_attrs = md.attributes(element_type='generator')
    bus_attrs = md.attributes(element_type='bus')
    branch_attrs = md.attributes(element_type='branch')

    inlet_branches_by_bus, outlet_branches_by_bus = \
        tx_utils.inlet_outlet_branches_by_bus(branches, buses)
    gens_by_bus = tx_utils.gens_by_bus(buses, gens)

    model = pe.ConcreteModel()

    ### declare (and fix) the loads at the buses
    bus_p_loads, _ = tx_utils.dict_of_bus_loads(buses, loads)

    libbus.declare_var_pl(model, bus_attrs['names'], initialize=bus_p_loads)
    model.pl.fix()

    ### declare the fixed shunts at the buses
    _, bus_gs_fixed_shunts = tx_utils.dict_of_bus_fixed_shunts(buses, shunts)

    ### declare the polar voltages
    va_bounds = {k: (-pi, pi) for k in bus_attrs['va']}
    libbus.declare_var_va(model, bus_attrs['names'], initialize=bus_attrs['va'],
                          bounds=va_bounds
                          )

    ### include the feasibility slack for the bus balances
    p_rhs_kwargs = {}
    penalty_expr = None
    if include_feasibility_slack:
        p_marginal_slack_penalty = _validate_and_extract_slack_penalty(md)        
        p_rhs_kwargs, penalty_expr = _include_feasibility_slack(model, bus_attrs['names'], bus_p_loads,
                                                                gens_by_bus, gen_attrs, p_marginal_slack_penalty)

    ### fix the reference bus
    ref_bus = md.data['system']['reference_bus']
    ref_angle = md.data['system']['reference_bus_angle']
    model.va[ref_bus].fix(radians(ref_angle))

    ### declare the generator real power
    pg_init = {k: (gen_attrs['p_min'][k] + gen_attrs['p_max'][k]) / 2.0 for k in gen_attrs['pg']}
    libgen.declare_var_pg(model, gen_attrs['names'], initialize=pg_init,
                          bounds=zip_items(gen_attrs['p_min'], gen_attrs['p_max'])
                          )

    ### declare the current flows in the branches
    vr_init = {k: bus_attrs['vm'][k] * pe.cos(bus_attrs['va'][k]) for k in bus_attrs['vm']}
    vj_init = {k: bus_attrs['vm'][k] * pe.sin(bus_attrs['va'][k]) for k in bus_attrs['vm']}
    p_max = {k: branches[k]['rating_long_term'] for k in branches.keys()}
    pf_bounds = dict()
    for k in branches.keys():
        if p_max[k] is None:
            pf_bounds[k] = (None, None)
        else:
            pf_bounds[k] = (-p_max[k],p_max[k])
    pf_init = dict()
    for branch_name, branch in branches.items():
        from_bus = branch['from_bus']
        to_bus = branch['to_bus']
        y_matrix = tx_calc.calculate_y_matrix_from_branch(branch)
        ifr_init = tx_calc.calculate_ifr(vr_init[from_bus], vj_init[from_bus], vr_init[to_bus],
                                         vj_init[to_bus], y_matrix)
        ifj_init = tx_calc.calculate_ifj(vr_init[from_bus], vj_init[from_bus], vr_init[to_bus],
                                         vj_init[to_bus], y_matrix)
        pf_init[branch_name] = tx_calc.calculate_p(ifr_init, ifj_init, vr_init[from_bus], vj_init[from_bus])

    libbranch.declare_var_pf(model=model,
                             index_set=branch_attrs['names'],
                             initialize=pf_init,
                             bounds=pf_bounds
                             )

    if dc_branches:
        dcpf_bounds = dict()
        for k, k_dict in dc_branches.items():
            kp_max = k_dict['rating_long_term']
            if kp_max is None:
                dcpf_bounds[k] = (None, None)
            else:
                dcpf_bounds[k] = (-kp_max, kp_max)
        libbranch.declare_var_dcpf(model=model,
                                   index_set=dc_branches.keys(),
                                   initialize=0.,
                                   bounds=dcpf_bounds,
                                  )
        dc_inlet_branches_by_bus, dc_outlet_branches_by_bus = \
                tx_utils.inlet_outlet_branches_by_bus(dc_branches, buses)
    else:
        dc_inlet_branches_by_bus = None
        dc_outlet_branches_by_bus = None


    ### declare the branch power flow approximation constraints
    libbranch.declare_eq_branch_power_btheta_approx(model=model,
                                                    index_set=branch_attrs['names'],
                                                    branches=branches
                                                    )

    ### declare the p balance
    libbus.declare_eq_p_balance_dc_approx(model=model,
                                          index_set=bus_attrs['names'],
                                          bus_p_loads=bus_p_loads,
                                          gens_by_bus=gens_by_bus,
                                          bus_gs_fixed_shunts=bus_gs_fixed_shunts,
                                          inlet_branches_by_bus=inlet_branches_by_bus,
                                          outlet_branches_by_bus=outlet_branches_by_bus,
                                          approximation_type=ApproximationType.BTHETA,
                                          dc_inlet_branches_by_bus=dc_inlet_branches_by_bus,
                                          dc_outlet_branches_by_bus=dc_outlet_branches_by_bus,
                                          **p_rhs_kwargs
                                          )

    ### declare the real power flow limits
    libbranch.declare_ineq_p_branch_thermal_lbub(model=model,
                                                 index_set=branch_attrs['names'],
                                                 branches=branches,
                                                 p_thermal_limits=p_max,
                                                 approximation_type=ApproximationType.BTHETA
                                                 )

    ### declare angle difference limits on interconnected buses
    if include_angle_diff_limits:
        libbranch.declare_ineq_angle_diff_branch_lbub(model=model,
                                                      index_set=branch_attrs['names'],
                                                      branches=branches,
                                                      coordinate_type=CoordinateType.POLAR
                                                      )

    ### declare the generator cost objective
    libgen.declare_expression_pgqg_operating_cost(model=model,
                                                  index_set=gen_attrs['names'],
                                                  p_costs=gen_attrs['p_cost']
                                                  )

    obj_expr = sum(model.pg_operating_cost[gen_name] for gen_name in model.pg_operating_cost)
    if include_feasibility_slack:
        obj_expr += penalty_expr

    model.obj = pe.Objective(expr=obj_expr)

    return model, md

def create_dense_lopf_model(model_data, include_feasibility_slack=False, base_point=BasePointType.SOLUTION, ptdf_options=None):

    ptdf_options = lpu.populate_default_ptdf_options(ptdf_options)

    baseMVA = model_data.data['system']['baseMVA']
    model_data.data['system']['load_mismatch_cost'] = 1000
    lpu.check_and_scale_ptdf_options(ptdf_options, baseMVA)

    md = model_data.clone_in_service()
    tx_utils.scale_ModelData_to_pu(md, inplace=True)

    gens = dict(md.elements(element_type='generator'))
    buses = dict(md.elements(element_type='bus'))
    branches = dict(md.elements(element_type='branch'))
    loads = dict(md.elements(element_type='load'))
    shunts = dict(md.elements(element_type='shunt'))

    dc_branches = dict(md.elements(element_type='dc_branch'))

    gen_attrs = md.attributes(element_type='generator')
    bus_attrs = md.attributes(element_type='bus')
    ## to keep things in order
    buses_idx = tuple(buses.keys())
    branches_idx = tuple(branches.keys())

    inlet_branches_by_bus, outlet_branches_by_bus = \
        tx_utils.inlet_outlet_branches_by_bus(branches, buses)
    gens_by_bus = tx_utils.gens_by_bus(buses, gens)

    model = pe.ConcreteModel()

    ### declare (and fix) the loads at the buses
    bus_p_loads, bus_q_loads = tx_utils.dict_of_bus_loads(buses, loads)

    libbus.declare_var_pl(model, buses_idx, initialize=bus_p_loads)
    libbus.declare_var_ql(model, buses_idx, initialize=bus_q_loads)
    model.pl.fix()
    model.ql.fix()

    ### declare the fixed shunts at the buses
    bus_bs_fixed_shunts, bus_gs_fixed_shunts = tx_utils.dict_of_bus_fixed_shunts(buses, shunts)

    ### declare the generator real and reactive power
    pg_init = {k: gen_attrs['pg'][k] for k in gens.keys()}
    qg_init = {k: gen_attrs['qg'][k] for k in gens.keys()}
    libgen.declare_var_pg(model, gen_attrs['names'], initialize=gen_attrs['pg'],
                          bounds=zip_items(gen_attrs['p_min'], gen_attrs['p_max'])
                          )
    libgen.declare_var_qg(model, gen_attrs['names'], initialize=gen_attrs['qg'],
                          bounds=zip_items(gen_attrs['q_min'], gen_attrs['q_max'])
                          )
    libbus.declare_var_vm(model, bus_attrs['names'], initialize=bus_attrs['vm'],
                          bounds=zip_items(bus_attrs['v_min'], bus_attrs['v_max']))
    libbus.declare_var_p_nw(model,buses_idx)
    libbus.declare_var_q_nw(model,buses_idx)

    #model.pg.fix()
    #model.qg.fix()
    #model.vm.fix()

    ### declare system and branch line losses
    libbranch.declare_var_ploss(model=model)
    libbranch.declare_var_qloss(model=model)
    libbranch.declare_var_pfl(model=model, index_set=branches_idx)
    libbranch.declare_var_qfl(model=model, index_set=branches_idx)

    ### include the feasibility slack for the system balance and other formulation settings
    p_rhs_kwargs = {}
    q_rhs_kwargs = {}
    if include_feasibility_slack:
        p_penalty = _validate_and_extract_slack_penalty(md)
        q_penalty = _validate_and_extract_slack_penalty(md)
        p_rhs_kwargs, q_rhs_kwargs, penalty_expr = _include_pq_feasibility_slack(model, bus_attrs['names'], bus_p_loads, bus_q_loads,
                                                                                 gens_by_bus, gen_attrs, p_penalty, q_penalty)
    p_rhs_kwargs['include_system_losses'] = True
    p_rhs_kwargs['linearize_shunts'] = True
    q_rhs_kwargs['include_system_losses'] = True
    q_rhs_kwargs['linearize_shunts'] = True

    if dc_branches:
        dcpf_bounds = dict()
        for k, k_dict in dc_branches.items():
            kp_max = k_dict['rating_long_term']
            if kp_max is None:
                dcpf_bounds[k] = (None, None)
            else:
                dcpf_bounds[k] = (-kp_max, kp_max)
        libbranch.declare_var_dcpf(model=model,
                                   index_set=dc_branches.keys(),
                                   initialize=0.,
                                   bounds=dcpf_bounds,
                                   )
        dc_inlet_branches_by_bus, dc_outlet_branches_by_bus = \
            tx_utils.inlet_outlet_branches_by_bus(dc_branches, buses)
    else:
        dc_inlet_branches_by_bus = None
        dc_outlet_branches_by_bus = None

    ### declare the power balance
    libbus.declare_eq_p_balance_lopf(model=model,
                                     index_set=buses_idx,
                                     bus_p_loads=bus_p_loads,
                                     gens_by_bus=gens_by_bus,
                                     bus_gs_fixed_shunts=bus_gs_fixed_shunts,
                                     vm_by_bus=bus_attrs['vm'],
                                     **p_rhs_kwargs
                                     )

    ### declare net withdraw constraints
    libbus.declare_eq_p_net_withdraw_at_bus(model=model,
                                            index_set=buses_idx,
                                            bus_p_loads=bus_p_loads,
                                            gens_by_bus=gens_by_bus,
                                            bus_gs_fixed_shunts=bus_gs_fixed_shunts,
                                            dc_inlet_branches_by_bus=dc_inlet_branches_by_bus,
                                            dc_outlet_branches_by_bus=dc_outlet_branches_by_bus,
                                            vm_by_bus=bus_attrs['vm'],
                                            **p_rhs_kwargs
                                            )
    libbus.declare_eq_q_net_withdraw_at_bus(model=model,
                                            index_set=buses_idx,
                                            bus_q_loads=bus_q_loads,
                                            gens_by_bus=gens_by_bus,
                                            bus_bs_fixed_shunts=bus_bs_fixed_shunts,
                                            vm_by_bus=bus_attrs['vm'],
                                            **q_rhs_kwargs)

    flow_bounds = dict()
    for bn,branch in branches.items():
        limit = branch['rating_long_term']
        flow_bounds[bn] = (-limit, limit)
    libbranch.declare_var_pf(model=model, index_set=branches_idx, bounds=flow_bounds)
    libbranch.declare_var_qf(model=model, index_set=branches_idx, bounds=flow_bounds)

    ### add apparent power flow expressions
    libbranch.declare_expr_sf(model=model, index_set=branches_idx)

    ## Do and store PTDF calculation
    reference_bus = md.data['system']['reference_bus']

    # ptdfobject
    PTDF = ptdf_utils.VirtualFDFpqMatrix(branches, buses, reference_bus, base_point, ptdf_options,
                                         branches_keys=branches_idx, buses_keys=buses_idx)

    model._PTDF = PTDF
    model._ptdf_options = ptdf_options

    if ptdf_options['lazy']:

        ## Add "full" system line loss constraints (will use loss factor residuals later)
        libbus.declare_eq_ploss_ptdf_approx(model=model,
                                            PTDF=PTDF,
                                            abs_ptdf_tol=ptdf_options['abs_ptdf_tol'],
                                            rel_ptdf_tol=ptdf_options['rel_ptdf_tol'],
                                            use_residuals=False
                                            )
        libbus.declare_eq_qloss_ptdf_approx(model=model,
                                            PTDF=PTDF,
                                            abs_ptdf_tol=ptdf_options['abs_ptdf_tol'],
                                            rel_ptdf_tol=ptdf_options['rel_ptdf_tol'],
                                            use_residuals=False
                                            )
        ### add "blank" thermal limits and power flow constraints
        libbranch.declare_ineq_pq_branch_thermal_bounds(model=model,
                                                        index_set=branches_idx,
                                                        branches=branches,
                                                        thermal_limits=None,
                                                        )
        libbranch.declare_eq_branch_power_ptdf_approx(model=model,
                                                      index_set=branches_idx,
                                                      PTDF=None
                                                      )
        libbranch.declare_eq_branch_power_qtdf_approx(model=model,
                                                      index_set=branches_idx,
                                                      PTDF=None
                                                      )
        libbranch.declare_eq_branch_loss_plf_approx(model=model,
                                                    index_set=branches_idx,
                                                    PTDF=None,
                                                    abs_ptdf_tol=ptdf_options['abs_ptdf_tol'],
                                                    rel_ptdf_tol=ptdf_options['rel_ptdf_tol']
                                                    )
        libbranch.declare_eq_branch_loss_qlf_approx(model=model,
                                                    index_set=branches_idx,
                                                    PTDF=None,
                                                    abs_ptdf_tol=ptdf_options['abs_ptdf_tol'],
                                                    rel_ptdf_tol=ptdf_options['rel_ptdf_tol']
                                                    )
        libbus.declare_eq_bus_vm_approx(model=model,
                                        index_set=buses_idx,
                                        PTDF=None
                                        )
        # add shunt buses to initial monitored set
        vm_initial_monitored_buses = [bn for bn,bs in bus_bs_fixed_shunts.items() if bs != 0]
        for bn in vm_initial_monitored_buses:
            buses[bn]['lazy'] = False

        ### add helpers for tracking monitored branches/buses
        lpu.add_monitored_flow_tracker(model)

        ### add initial branches/buses to monitored set
        lpu.add_initial_monitored_branches(model, branches, branches_idx, ptdf_options, PTDF)
        lpu.add_initial_monitored_buses(model, buses, buses_idx, ptdf_options, PTDF)

    else:
        #add "dummy" system loss constraints to take sum of all branch losses
        libbus.declare_eq_ploss_sum_of_pfl(model=model, index_set=branches_idx)
        libbus.declare_eq_qloss_sum_of_qfl(model=model, index_set=branches_idx)

        thermal_limits = {k: branches[k]['rating_long_term'] for k in branches.keys()}
        ## add all the constraints
        ### declare the branch power flow approximation constraints
        libbranch.declare_eq_branch_power_ptdf_approx(model=model,
                                                      index_set=branches_idx,
                                                      PTDF=PTDF,
                                                      abs_ptdf_tol=ptdf_options['abs_ptdf_tol'],
                                                      rel_ptdf_tol=ptdf_options['rel_ptdf_tol'],
                                                      )
        libbranch.declare_eq_branch_power_qtdf_approx(model=model,
                                                      index_set=branches_idx,
                                                      PTDF=PTDF,
                                                      abs_ptdf_tol=ptdf_options['abs_ptdf_tol'],
                                                      rel_ptdf_tol=ptdf_options['rel_ptdf_tol'],
                                                      )
        libbranch.declare_eq_branch_loss_plf_approx(model=model,
                                                    PTDF=PTDF,
                                                    abs_ptdf_tol=ptdf_options['abs_ptdf_tol'],
                                                    rel_ptdf_tol=ptdf_options['rel_ptdf_tol']
                                                    )
        libbranch.declare_eq_branch_loss_qlf_approx(model=model,
                                                    PTDF=PTDF,
                                                    abs_ptdf_tol=ptdf_options['abs_ptdf_tol'],
                                                    rel_ptdf_tol=ptdf_options['rel_ptdf_tol']
                                                    )
        libbus.declare_eq_bus_vm_approx(model=model,
                                        index_set=buses_idx,
                                        PTDF=PTDF,
                                        abs_ptdf_tol=ptdf_options['abs_ptdf_tol'],
                                        rel_ptdf_tol=ptdf_options['rel_ptdf_tol'],
                                        )

        ### add all the limits
        libbranch.declare_ineq_pq_branch_thermal_bounds(model=model,
                                                        index_set=branches_idx,
                                                        branches=branches,
                                                        thermal_limits=thermal_limits,
                                                        )

    # reactive base point deviation penalty
    #q_dev_bounds = {g: (0, gen['q_max']-gen['q_min']) for g,gen in gens.items()}
    q_dev_bounds = {g: (0, np.inf) for g in gen_attrs['names']}
    decl.declare_var('q_pos', model=model, index_set=gen_attrs['names'], bounds=q_dev_bounds)
    decl.declare_var('q_neg', model=model, index_set=gen_attrs['names'], bounds=q_dev_bounds)
    libgen.declare_eq_q_fdf_deviation(model=model, index_set=gen_attrs['names'],gens=gens)

    ### declare the generator cost objective
    libgen.declare_expression_pgqg_operating_cost(model=model,
                                                  index_set=gen_attrs['names'],
                                                  p_costs=gen_attrs['p_cost'],
                                                  q_costs='penalty'
                                                  )

    obj_expr = sum(model.pg_operating_cost[gen_name] for gen_name in model.pg_operating_cost)
    obj_expr += sum(model.qg_operating_cost[gen_name] for gen_name in model.qg_operating_cost)
    if include_feasibility_slack:
        obj_expr += penalty_expr

    model.obj = pe.Objective(expr=obj_expr)

    return model, md

def create_compact_lopf_model(model_data, include_feasibility_slack=False, base_point=BasePointType.SOLUTION, ptdf_options=None):

    ptdf_options = lpu.populate_default_ptdf_options(ptdf_options)

    baseMVA = model_data.data['system']['baseMVA']
    model_data.data['system']['load_mismatch_cost'] = 1000
    lpu.check_and_scale_ptdf_options(ptdf_options, baseMVA)

    md = model_data.clone_in_service()
    tx_utils.scale_ModelData_to_pu(md, inplace=True)

    gens = dict(md.elements(element_type='generator'))
    buses = dict(md.elements(element_type='bus'))
    branches = dict(md.elements(element_type='branch'))
    loads = dict(md.elements(element_type='load'))
    shunts = dict(md.elements(element_type='shunt'))

    dc_branches = dict(md.elements(element_type='dc_branch'))

    gen_attrs = md.attributes(element_type='generator')
    bus_attrs = md.attributes(element_type='bus')
    ## to keep things in order
    buses_idx = tuple(buses.keys())
    branches_idx = tuple(branches.keys())

    inlet_branches_by_bus, outlet_branches_by_bus = \
        tx_utils.inlet_outlet_branches_by_bus(branches, buses)
    gens_by_bus = tx_utils.gens_by_bus(buses, gens)

    model = pe.ConcreteModel()

    ### declare (and fix) the loads at the buses
    bus_p_loads, bus_q_loads = tx_utils.dict_of_bus_loads(buses, loads)

    libbus.declare_var_pl(model, buses_idx, initialize=bus_p_loads)
    libbus.declare_var_ql(model, buses_idx, initialize=bus_q_loads)
    model.pl.fix()
    model.ql.fix()

    ### declare the fixed shunts at the buses
    bus_bs_fixed_shunts, bus_gs_fixed_shunts = tx_utils.dict_of_bus_fixed_shunts(buses, shunts)

    ### declare the generator real and reactive power
    pg_init = {k: gen_attrs['pg'][k] for k in gens.keys()}
    qg_init = {k: gen_attrs['qg'][k] for k in gens.keys()}
    libgen.declare_var_pg(model, gen_attrs['names'], initialize=gen_attrs['pg'],
                          bounds=zip_items(gen_attrs['p_min'], gen_attrs['p_max'])
                          )
    libgen.declare_var_qg(model, gen_attrs['names'], initialize=gen_attrs['qg'],
                          bounds=zip_items(gen_attrs['q_min'], gen_attrs['q_max'])
                          )
    libbus.declare_var_vm(model, bus_attrs['names'], initialize=bus_attrs['vm'],
                          bounds=zip_items(bus_attrs['v_min'], bus_attrs['v_max']))
    libbus.declare_var_p_nw(model,buses_idx)
    libbus.declare_var_q_nw(model,buses_idx)

    #model.pg.fix()
    #model.qg.fix()
    #model.vm.fix()

    ### declare system real power line losses
    libbranch.declare_var_ploss(model=model)
    libbranch.declare_var_qloss(model=model)

    ### include the feasibility slack for the system balance and other formulation settings
    p_rhs_kwargs = {}
    q_rhs_kwargs = {}
    if include_feasibility_slack:
        p_penalty = _validate_and_extract_slack_penalty(md)
        q_penalty = _validate_and_extract_slack_penalty(md)
        p_rhs_kwargs, q_rhs_kwargs, penalty_expr = _include_pq_feasibility_slack(model, bus_attrs['names'], bus_p_loads, bus_q_loads,
                                                                                 gens_by_bus, gen_attrs, p_penalty, q_penalty)
    p_rhs_kwargs['include_system_losses'] = True
    p_rhs_kwargs['linearize_shunts'] = True
    q_rhs_kwargs['include_system_losses'] = True
    q_rhs_kwargs['linearize_shunts'] = True

    if dc_branches:
        dcpf_bounds = dict()
        for k, k_dict in dc_branches.items():
            kp_max = k_dict['rating_long_term']
            if kp_max is None:
                dcpf_bounds[k] = (None, None)
            else:
                dcpf_bounds[k] = (-kp_max, kp_max)
        libbranch.declare_var_dcpf(model=model,
                                   index_set=dc_branches.keys(),
                                   initialize=0.,
                                   bounds=dcpf_bounds,
                                   )
        dc_inlet_branches_by_bus, dc_outlet_branches_by_bus = \
            tx_utils.inlet_outlet_branches_by_bus(dc_branches, buses)
    else:
        dc_inlet_branches_by_bus = None
        dc_outlet_branches_by_bus = None

    ### declare the power balance
    libbus.declare_eq_p_balance_lopf(model=model,
                                     index_set=buses_idx,
                                     bus_p_loads=bus_p_loads,
                                     gens_by_bus=gens_by_bus,
                                     bus_gs_fixed_shunts=bus_gs_fixed_shunts,
                                     vm_by_bus=bus_attrs['vm'],
                                     **p_rhs_kwargs
                                     )

    ### declare net withdraw constraints
    libbus.declare_eq_p_net_withdraw_at_bus(model=model,
                                            index_set=buses_idx,
                                            bus_p_loads=bus_p_loads,
                                            gens_by_bus=gens_by_bus,
                                            bus_gs_fixed_shunts=bus_gs_fixed_shunts,
                                            dc_inlet_branches_by_bus=dc_inlet_branches_by_bus,
                                            dc_outlet_branches_by_bus=dc_outlet_branches_by_bus,
                                            vm_by_bus=bus_attrs['vm'],
                                            **p_rhs_kwargs
                                            )
    libbus.declare_eq_q_net_withdraw_at_bus(model=model,
                                            index_set=buses_idx,
                                            bus_q_loads=bus_q_loads,
                                            gens_by_bus=gens_by_bus,
                                            bus_bs_fixed_shunts=bus_bs_fixed_shunts,
                                            vm_by_bus=bus_attrs['vm'],
                                            **q_rhs_kwargs)

    flow_bounds = dict()
    for bn,branch in branches.items():
        limit = branch['rating_long_term']
        flow_bounds[bn] = (-limit, limit)
    libbranch.declare_var_pf(model=model, index_set=branches_idx, bounds=flow_bounds)
    libbranch.declare_var_qf(model=model, index_set=branches_idx, bounds=flow_bounds)

    ### add apparent power flow expressions
    libbranch.declare_expr_sf(model=model, index_set=branches_idx)

    ## Do and store PTDF calculation
    reference_bus = md.data['system']['reference_bus']

    # ptdfobject
    PTDF = ptdf_utils.VirtualFDFpqMatrix(branches, buses, reference_bus, base_point, ptdf_options,
                                         branches_keys=branches_idx, buses_keys=buses_idx)

    model._PTDF = PTDF
    model._ptdf_options = ptdf_options

    ## Ploss constraints
    libbus.declare_eq_ploss_ptdf_approx(model=model,
                                        PTDF=PTDF,
                                        abs_ptdf_tol=ptdf_options['abs_ptdf_tol'],
                                        rel_ptdf_tol=ptdf_options['rel_ptdf_tol'],
                                        )
    libbus.declare_eq_qloss_ptdf_approx(model=model,
                                        PTDF=PTDF,
                                        abs_ptdf_tol=ptdf_options['abs_ptdf_tol'],
                                        rel_ptdf_tol=ptdf_options['rel_ptdf_tol'],
                                        )

    if ptdf_options['lazy']:

        ### add "blank" thermal limits and power flow constraints
        libbranch.declare_ineq_pq_branch_thermal_bounds(model=model,
                                                        index_set=branches_idx,
                                                        branches=branches,
                                                        thermal_limits=None,
                                                        )
        libbranch.declare_eq_branch_power_ptdf_approx(model=model,
                                                      index_set=branches_idx,
                                                      PTDF=None
                                                      )
        libbranch.declare_eq_branch_power_qtdf_approx(model=model,
                                                      index_set=branches_idx,
                                                      PTDF=None
                                                      )
        libbus.declare_eq_bus_vm_approx(model=model,
                                        index_set=buses_idx,
                                        PTDF=None
                                        )
        # add shunt buses to initial monitored set
        vm_initial_monitored_buses = [bn for bn,bs in bus_bs_fixed_shunts.items() if bs != 0]
        for bn in vm_initial_monitored_buses:
            buses[bn]['lazy'] = False

        ### add helpers for tracking monitored branches/buses
        lpu.add_monitored_flow_tracker(model)

        ### add initial branches/buses to monitored set
        lpu.add_initial_monitored_branches(model, branches, branches_idx, ptdf_options, PTDF)
        lpu.add_initial_monitored_buses(model, buses, buses_idx, ptdf_options, PTDF)

    else:
        thermal_limits = {k: branches[k]['rating_long_term'] for k in branches.keys()}
        ## add all the constraints
        ### declare the branch power flow approximation constraints
        libbranch.declare_eq_branch_power_ptdf_approx(model=model,
                                                      index_set=branches_idx,
                                                      PTDF=PTDF,
                                                      abs_ptdf_tol=ptdf_options['abs_ptdf_tol'],
                                                      rel_ptdf_tol=ptdf_options['rel_ptdf_tol'],
                                                      )
        libbranch.declare_eq_branch_power_qtdf_approx(model=model,
                                                      index_set=branches_idx,
                                                      PTDF=PTDF,
                                                      abs_ptdf_tol=ptdf_options['abs_ptdf_tol'],
                                                      rel_ptdf_tol=ptdf_options['rel_ptdf_tol'],
                                                      )
        libbus.declare_eq_bus_vm_approx(model=model,
                                        index_set=buses_idx,
                                        PTDF=PTDF,
                                        abs_ptdf_tol=ptdf_options['abs_ptdf_tol'],
                                        rel_ptdf_tol=ptdf_options['rel_ptdf_tol'],
                                        )

        ### add all the limits
        libbranch.declare_ineq_pq_branch_thermal_bounds(model=model,
                                                        index_set=branches_idx,
                                                        branches=branches,
                                                        thermal_limits=thermal_limits,
                                                        )

    # reactive base point deviation penalty
    #q_dev_bounds = {g: (0, gen['q_max']-gen['q_min']) for g,gen in gens.items()}
    q_dev_bounds = {g: (0, np.inf) for g in gen_attrs['names']}
    decl.declare_var('q_pos', model=model, index_set=gen_attrs['names'], bounds=q_dev_bounds)
    decl.declare_var('q_neg', model=model, index_set=gen_attrs['names'], bounds=q_dev_bounds)
    libgen.declare_eq_q_fdf_deviation(model=model, index_set=gen_attrs['names'],gens=gens)

    ### declare the generator cost objective
    libgen.declare_expression_pgqg_operating_cost(model=model,
                                                  index_set=gen_attrs['names'],
                                                  p_costs=gen_attrs['p_cost'],
                                                  q_costs='penalty'
                                                  )

    obj_expr = sum(model.pg_operating_cost[gen_name] for gen_name in model.pg_operating_cost)
    obj_expr += sum(model.qg_operating_cost[gen_name] for gen_name in model.qg_operating_cost)
    if include_feasibility_slack:
        obj_expr += penalty_expr

    model.obj = pe.Objective(expr=obj_expr)

    return model, md

def create_p_lopf_model(model_data, include_feasibility_slack=False, base_point=BasePointType.SOLUTION, ptdf_options=None):

    ptdf_options = lpu.populate_default_ptdf_options(ptdf_options)

    baseMVA = model_data.data['system']['baseMVA']
    #model_data.data['system']['load_mismatch_cost'] = 1000
    lpu.check_and_scale_ptdf_options(ptdf_options, baseMVA)
    
    md = model_data.clone_in_service()
    tx_utils.scale_ModelData_to_pu(md, inplace = True)

    gens = dict(md.elements(element_type='generator'))
    buses = dict(md.elements(element_type='bus'))
    branches = dict(md.elements(element_type='branch'))
    loads = dict(md.elements(element_type='load'))
    shunts = dict(md.elements(element_type='shunt'))

    dc_branches = dict(md.elements(element_type='dc_branch'))

    gen_attrs = md.attributes(element_type='generator')
    bus_attrs = md.attributes(element_type='bus')
    ## to keep things in order
    buses_idx = tuple(buses.keys())
    branches_idx = tuple(branches.keys())

    inlet_branches_by_bus, outlet_branches_by_bus = \
        tx_utils.inlet_outlet_branches_by_bus(branches, buses)
    gens_by_bus = tx_utils.gens_by_bus(buses, gens)

    model = pe.ConcreteModel()

    ### declare (and fix) the loads at the buses
    bus_p_loads, _ = tx_utils.dict_of_bus_loads(buses, loads)

    libbus.declare_var_pl(model, buses_idx, initialize=bus_p_loads)
    model.pl.fix()

    ### declare the fixed shunts at the buses
    _, bus_gs_fixed_shunts = tx_utils.dict_of_bus_fixed_shunts(buses, shunts)

    ### declare the generator real power
    pg_init = {k: gen_attrs['pg'][k] for k in gens.keys()}
    libgen.declare_var_pg(model, gen_attrs['names'], initialize=gen_attrs['pg'],
                          bounds=zip_items(gen_attrs['p_min'], gen_attrs['p_max'])
                          )
    libbus.declare_var_p_nw(model,buses_idx)

    #libbus.declare_var_vm(model, bus_attrs['names'], initialize=bus_attrs['vm'],
    #                      bounds=zip_items(bus_attrs['v_min'], bus_attrs['v_max']))

    ### declare system real power line losses
    libbranch.declare_var_ploss(model=model)

    ### include the feasibility slack for the system balance and other formulation settings
    p_rhs_kwargs = {}
    if include_feasibility_slack:
        p_marginal_slack_penalty = _validate_and_extract_slack_penalty(md)                
        p_rhs_kwargs, penalty_expr = _include_feasibility_slack(model, bus_attrs['names'], bus_p_loads,
                                                                gens_by_bus, gen_attrs, p_marginal_slack_penalty)
    p_rhs_kwargs['include_system_losses'] = True
    p_rhs_kwargs['linearize_shunts'] = False

    if dc_branches:
        dcpf_bounds = dict()
        for k, k_dict in dc_branches.items():
            kp_max = k_dict['rating_long_term']
            if kp_max is None:
                dcpf_bounds[k] = (None, None)
            else:
                dcpf_bounds[k] = (-kp_max, kp_max)
        libbranch.declare_var_dcpf(model=model,
                                   index_set=dc_branches.keys(),
                                   initialize=0.,
                                   bounds=dcpf_bounds,
                                  )
        dc_inlet_branches_by_bus, dc_outlet_branches_by_bus = \
                tx_utils.inlet_outlet_branches_by_bus(dc_branches, buses)
    else:
        dc_inlet_branches_by_bus = None
        dc_outlet_branches_by_bus = None

    ### declare the p balance
    libbus.declare_eq_p_balance_lopf(model=model,
                                   index_set=buses_idx,
                                   bus_p_loads=bus_p_loads,
                                   gens_by_bus=gens_by_bus,
                                   bus_gs_fixed_shunts=bus_gs_fixed_shunts,
                                   vm_by_bus=bus_attrs['vm'],
                                   **p_rhs_kwargs
                                   )

    ### declare net withdraw constraints
    libbus.declare_eq_p_net_withdraw_at_bus(model=model,
                                            index_set=buses_idx,
                                            bus_p_loads=bus_p_loads,
                                            gens_by_bus=gens_by_bus,
                                            bus_gs_fixed_shunts=bus_gs_fixed_shunts,
                                            dc_inlet_branches_by_bus=dc_inlet_branches_by_bus,
                                            dc_outlet_branches_by_bus=dc_outlet_branches_by_bus,
                                            vm_by_bus=bus_attrs['vm'],
                                            **p_rhs_kwargs
                                            )
    
    ### add "blank" power flow expressions
    libbranch.declare_expr_pf(model=model,
                              index_set=branches_idx,
                              )

    ## Do and store PTDF calculation
    reference_bus = md.data['system']['reference_bus']

    #ptdfobject
    PTDF = ptdf_utils.VirtualFDFpMatrix(branches, buses, reference_bus, base_point, ptdf_options, branches_keys=branches_idx, buses_keys=buses_idx)
    PTDF.add_q_correction()

    model._PTDF = PTDF
    model._ptdf_options = ptdf_options

    ## Ploss constraints
    libbus.declare_eq_ploss_ptdf_approx(model=model,
                                        PTDF=PTDF,
                                        abs_ptdf_tol=ptdf_options['abs_ptdf_tol'],
                                        rel_ptdf_tol=ptdf_options['rel_ptdf_tol'],
                                        )

    if ptdf_options['lazy']:

        ### add "blank" real power flow limits
        libbranch.declare_ineq_p_branch_thermal_bounds(model=model,
                                                       index_set=branches_idx,
                                                       branches=branches,
                                                       p_thermal_limits=None,
                                                       approximation_type=None,
                                                       )

        ### add helpers for tracking monitored branches
        lpu.add_monitored_flow_tracker(model)

        ### add initial branches to monitored set
        lpu.add_initial_monitored_branches(model, branches, branches_idx, ptdf_options, PTDF)

    else:
        p_max = {k: branches[k]['rating_long_term'] - PTDF._q_correction[k] for k in branches.keys()}
        ## add all the constraints
        ### declare the branch power flow approximation constraints
        libbranch.declare_eq_branch_power_ptdf_approx(model=model,
                                                      index_set=branches_idx,
                                                      PTDF=PTDF,
                                                      abs_ptdf_tol=ptdf_options['abs_ptdf_tol'],
                                                      rel_ptdf_tol=ptdf_options['rel_ptdf_tol'],
                                                      )

        ### add all the limits
        libbranch.declare_ineq_p_branch_thermal_lbub(model=model,
                                                     index_set=branches_idx,
                                                     branches=branches,
                                                     p_thermal_limits=p_max,
                                                     approximation_type=ApproximationType.PTDF,
                                                     )

    ### declare the generator cost objective
    libgen.declare_expression_pgqg_operating_cost(model=model,
                                                  index_set=gen_attrs['names'],
                                                  p_costs=gen_attrs['p_cost']
                                                  )

    obj_expr = sum(model.pg_operating_cost[gen_name] for gen_name in model.pg_operating_cost)
    if include_feasibility_slack:
        obj_expr += penalty_expr

    model.obj = pe.Objective(expr=obj_expr)


    return model, md

def _lazy_ptdf_lopf_model_solve_loop(m, md, solver, solver_tee=True, symbolic_solver_labels=False, iteration_limit=100000):
    '''
    The lazy PTDF LOPF solver loop. This function iteratively
    adds violated transmission constraints until either the result is
    transmission feasible or we're tracking every violated constraint
    in the model

    Parameters
    ----------
    m : pyomo.environ.ConcreteModel
        An egret DCOPF model with no transmission constraints
    md : egret.data.ModelData
        An egret ModelData object
    solver : pyomo.opt.solver
        A pyomo solver object
    solver_tee : bool (optional)
        For displaying the solver log (default is True)
    symbolic_solver_labels : bool (optional)
        Use symbolic solver labels when writing to the solver (default is False)
    iteration_limit : int (optional)
        Number of iterations before a hard termination (default is 100000)

    Returns
    -------
    egret.common.lazy_ptdf_utils.LazyPTDFTerminationCondition : the termination status
    pyomo.opt.results.SolverResults : The results object from the pyomo solver
    int : The number of iterations before termination

    '''
    from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
    from egret.common.solver_interface import _solve_model

    PTDF = m._PTDF

    ptdf_options = m._ptdf_options

    persistent_solver = isinstance(solver, PersistentSolver)
    mw_only = PTDF._mw_only
    vm_viol_num = 0
    mon_vm_viol_num = 0

    for i in range(iteration_limit):

        flows, viol_num, mon_viol_num, viol_lazy \
                = lpu.check_violations(m, md, PTDF, ptdf_options['max_violations_per_iteration'])
        if not mw_only:
            flows, vm_viol_num, mon_vm_viol_num, vm_viol_lazy \
                = lpu.check_vm_violations(m, md, PTDF, ptdf_options['max_vm_violations_per_iteration'], flows=flows)

        iter_status_str = "iteration {0}, found {1} violation(s)".format(i,viol_num)
        if mon_viol_num:
            iter_status_str += ", {} of which are already monitored".format(mon_viol_num)
        if not mw_only and vm_viol_num > 0:
            iter_status_str += ", and {} voltage violation(s)".format(vm_viol_num)
            if mon_vm_viol_num:
                iter_status_str += ", {} already monitored".format(mon_vm_viol_num)

        logger.info(iter_status_str)

        if viol_num + vm_viol_num <= 0:
            ## in this case, there are no violations!
            ## load the duals now too, if we're using a persistent solver
            if persistent_solver:
                solver.load_duals()
            return lpu.LazyPTDFTerminationCondition.NORMAL, i

        elif viol_num + vm_viol_num == mon_viol_num + mon_vm_viol_num:
            logger.warning('WARNING: Terminating with monitored violations! Result is not transmission feasible.')
            if persistent_solver:
                solver.load_duals()
            return lpu.LazyPTDFTerminationCondition.FLOW_VIOLATION, i

        lpu.add_violations(viol_lazy, flows, m, md, solver, ptdf_options, PTDF)
        if not mw_only:
            lpu.add_vm_violations(vm_viol_lazy, flows, m, md, solver, ptdf_options, PTDF)
        total_flow_constr_added = len(viol_lazy)
        add_constr_message =  "iteration {0}, added {1} flow constraint(s)".format(i,total_flow_constr_added)
        if not mw_only:
            add_constr_message += ", added {} voltage constraint(s)".format(len(vm_viol_lazy))
        logger.info(add_constr_message)

        if persistent_solver:
            m, results, solver = _solve_model(m, solver, solver_tee=solver_tee, return_solver=True, vars_to_load=[], set_instance=False)
            solver.load_vars()
        else:
            m, results, solver = _solve_model(m, solver, solver_tee=solver_tee, return_solver=True)

    else: # we hit the iteration limit
        logger.warning('WARNING: Exiting on maximum iterations for lazy PTDF model. Result is not transmission feasible.')
        if persistent_solver:
            solver.load_duals()
        return lpu.LazyPTDFTerminationCondition.ITERATION_LIMIT, i


def solve_lopf(model_data,
                solver,
                timelimit = None,
                solver_tee = True,
                symbolic_solver_labels = False,
                options = None,
                lopf_model_generator = create_sparse_lopf_model,
                return_model = False,
                return_results = False,
                **kwargs):
    '''
    Create and solve a new dcopf model

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
    dcopf_model_generator : function (optional)
        Function for generating the dcopf model. Default is
        egret.models.dcopf.create_btheta_dcopf_model
    return_model : bool (optional)
        If True, returns the pyomo model object
    return_results : bool (optional)
        If True, returns the pyomo results object
    kwargs : dictionary (optional)
        Additional arguments for building model
    '''

    import pyomo.environ as pe
    import pyomo.opt as po
    import time
    from pyomo.environ import value
    from egret.common.solver_interface import _solve_model
    from egret.model_library.transmission.tx_utils import \
        scale_ModelData_to_pu, unscale_ModelData_to_pu
    from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver

    m, md = lopf_model_generator(model_data, **kwargs)

    m.dual = pe.Suffix(direction=pe.Suffix.IMPORT)

    ## flag for persistent solver
    persistent_solver = isinstance(solver, PersistentSolver) or 'persistent' in solver

    m, results, solver = _solve_model(m,solver,timelimit=timelimit,solver_tee=solver_tee,
                              symbolic_solver_labels=symbolic_solver_labels,solver_options=options, return_solver=True)

    if persistent_solver or solver.name=='gurobi_direct':
        init_solve_time = results.Solver[0]['Wallclock time']
    else:
        init_solve_time = results.Solver.Time
    start_loop_time = time.time()

    lazy_solve_loop = hasattr(m, '_ptdf_options') and m._ptdf_options['lazy']
    if lazy_solve_loop:
        iter_limit = m._ptdf_options['iteration_limit']
        term_cond, iterations = _lazy_ptdf_lopf_model_solve_loop(m, md, solver, solver_tee=solver_tee, symbolic_solver_labels=symbolic_solver_labels,iteration_limit=iter_limit)

    loop_time = time.time() - start_loop_time
    total_time = init_solve_time + loop_time

    # save results data to ModelData object
    gens = dict(md.elements(element_type='generator'))
    buses = dict(md.elements(element_type='bus'))
    branches = dict(md.elements(element_type='branch'))

    dc_branches = dict(md.elements(element_type='dc_branch'))

    md.data['system']['total_cost'] = value(m.obj)
    if not hasattr(md,'results'):
        md.data['results'] = dict()
    md.data['results']['time'] = total_time
    md.data['results']['#_cons'] = results.Problem[0]['Number of constraints']
    md.data['results']['#_vars'] = results.Problem[0]['Number of variables']
    md.data['results']['#_nz'] = results.Problem[0]['Number of nonzeros']
    md.data['results']['termination'] = results.solver.termination_condition.__str__()
    if lazy_solve_loop:
        md.data['results']['iterations'] = iterations

    for g,g_dict in gens.items():
        g_dict['pg'] = value(m.pg[g])

    ## calculate the power flows from our PTDF matrix for maximum precision
    ## calculate the LMPC (LMP congestion) using numpy
    if lopf_model_generator == create_sparse_lopf_model:
        mw_only = False
        for k, k_dict in branches.items():
            k_dict['pf'] = value(m.pf[k])
            k_dict['qf'] = value(m.qf[k])
        for b, b_dict in buses.items():
            b_dict['va'] = value(m.va[b])
            b_dict['vm'] = value(m.vm[b])
    else:
        PTDF = m._PTDF
        mw_only = PTDF._mw_only
        PFV, _, VA = PTDF.calculate_PFV(m)
        PLV = PTDF.calculate_PLV(m, VA=VA)
        branches_idx = PTDF.branches_keys
        buses_idx = PTDF.buses_keys
        for i,bn in enumerate(branches_idx):
            branches[bn]['pf'] = PFV[i]
            branches[bn]['pfl'] = PLV[i]
        for i,b in enumerate(buses_idx):
            buses[b]['va'] = VA[i]

        if not mw_only:
            QFV, VM = PTDF.calculate_QFV(m)
            QLV = PTDF.calculate_QLV(m, VM=VM)
            for g,g_dict in gens.items():
                g_dict['qg'] = value(m.qg[g])
            for i,bn in enumerate(branches_idx):
                branches[bn]['qf'] = QFV[i]
                branches[bn]['qfl'] = QLV[i]
            for i,b in enumerate(buses_idx):
                buses[b]['vm'] = VM[i]

    if lopf_model_generator in [create_p_lopf_model,create_compact_lopf_model,create_dense_lopf_model]:
        if hasattr(m, 'p_load_shed'):
            md.data['system']['p_balance_violation'] = value(m.p_load_shed) - value(m.p_over_generation)
        buses_idx = PTDF.buses_keys
        LMP = PTDF.calculate_LMP(m, m.dual, m.eq_p_balance)
        for i,b in enumerate(buses_idx):
            b_dict = buses[b]
            b_dict['lmp'] = LMP[i]
            b_dict['pl'] = value(m.pl[b])
            b_dict['va'] = VA[i]
    elif lopf_model_generator == create_sparse_lopf_model:
        for b,b_dict in buses.items():
            if hasattr(m, 'p_load_shed'):
                b_dict['p_balance_violation'] = value(m.p_load_shed[b]) - value(m.p_over_generation[b])
            b_dict['pl'] = value(m.pl[b])
            b_dict['lmp'] = value(m.dual[m.eq_p_balance[b]])
            b_dict['va'] = value(m.va[b])
    else:
        raise Exception("Unrecognized lopf_model_generator {}".format(lopf_model_generator))

    for k, k_dict in dc_branches.items():
        k_dict['pf'] = value(m.dcpf[k])

    unscale_ModelData_to_pu(md, inplace=True)

    if return_model and return_results:
        return md, m, results
    elif return_model:
        return md, m
    elif return_results:
        return md, results
    return md

def check_instance_feasibility(instance, tolerance, active_only=True):
    infeasibilities = list()

    for con in instance.component_data_objects(pe.Constraint, descend_into=True, sort=True):
        if active_only == False or con.active == True:
            resid = compute_constraint_resid(con)
            if (resid > tolerance):
                infeasibilities.append(constraint_resid_to_string(con.getname(True), con, resid))

    for var in instance.component_data_objects(pe.Var, descend_into=True, sort=True):
        lb = var.lb
        ub = var.ub

        if (ub is not None and lb is not None and ub < lb):
            infeasibility_found = True
            infeasibilities.append('Var: {0} has an upper bound ({1}) that is smaller than its lower bound ({2})'.format(
                var.getname(True), ub, lb))
        if (ub is not None and value(var) > ub):
            infeasibility_found = True
            infeasibilities.append('Var: {0} has an value ({1} that is greater than its upper bound ({2})'.format(
                var.getname(True), value(var), ub))
        if (lb and value(var) < lb):
            infeasibility_found = True
            infeasibilities.append('Var: {0} has an value ({1}) that is less than its lower bound ({2})'.format(
                var.getname(True), value(var), lb))

    if len(infeasibilities) > 0:
        print("*** Infeasibilities found in check_instance_feasibility")
        for s in infeasibilities:
            print(s)
        print("***")

    return len(infeasibilities) == 0

def compute_constraint_resid(con):
    bodyval = value(con.body)
    upper_resid = 0
    if con.upper is not None:
        upper_resid = max(0, bodyval - value(con.upper))
    lower_resid = 0
    if con.lower is not None:
        lower_resid = max(0, value(con.lower) - bodyval)
    return  max(upper_resid, lower_resid)

def constraint_resid_to_string(name, con, resid):
    if con.lower is None and con.upper is None:
        return '{0:10.4g} | {2:10s} <= {3:10.4g} <= {4:10s} : {1}'.format(resid, name, '-', value(con.body), '-')
    elif con.lower is None:
        return '{0:10.4g} | {2:10s} <= {3:10.4g} <= {4:10.4g} : {1}'.format(resid, name, '-', value(con.body), value(con.upper))
    elif con.upper is None:
        return '{0:10.4g} | {2:10.4} <= {3:10.4g} <= {4:10s} : {1}'.format(resid, name, value(con.lower), value(con.body), '-')
    else:
        return '{0:10.4g} | {2:10.4} <= {3:10.4g} <= {4:10.4g} : {1}'.format(resid, name, value(con.lower), value(con.body), value(con.upper))




if __name__ == '__main__':
    import os
    import egret.data.test_utils as tu
    from egret.parsers.matpower_parser import create_ModelData
    from egret.models.acopf import solve_acopf
    from egret.data.lopf_utils import case_names

    path = os.path.dirname(__file__)
    print(path)
    #filename = 'pglib_opf_case5_pjm.m'
    #filename = 'pglib_opf_case30000_goc.m'
    filename = case_names[15] + '.m'
    test_case = os.path.join(path, '../thirdparty/pglib-opf-master/', filename)
    model_data = create_ModelData(test_case)
    print(filename)

    #lopf_model = create_p_lopf_model
    #lopf_model = create_compact_lopf_model
    lopf_model = create_dense_lopf_model
    #lopf_model = create_s_lopf_model


    md_ac, m_ac, results = solve_acopf(model_data, "ipopt",return_model=True, return_results=True, solver_tee=True)
    print('Solved ACOPF base point.')
    print('..Total cost: ${}'.format(md_ac.data['system']['total_cost']))
    sd = tu.update_solution_dicts(md_ac, name='acopf')
    print(results['Solver'])
    print(filename)

    kwargs = {'ptdf_options': {'lazy':True}}
    #logger.setLevel(logging.INFO)
    md_fdf, m_fdf, results = solve_lopf(md_ac, "gurobi_persistent", lopf_model_generator=lopf_model, solver_tee=False,
                                             return_model=True, return_results=True, **kwargs)
    print('..Total cost: ${}'.format(md_fdf.data['system']['total_cost']))
    sd = tu.update_solution_dicts(md_fdf, name='lopf', solution_dict=sd)
    tu.display_solution_dicts(sd, N=5)
    print(md_fdf.data['results'])
    print(filename)

