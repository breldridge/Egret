#  ___________________________________________________________________________
#
#  EGRET: Electrical Grid Research and Engineering Tools
#  Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
This module contains the declarations for the modeling components
typically used for buses (including loads and shunts)
"""
import pyomo.environ as pe
import egret.model_library.decl as decl
from pyomo.core.util import quicksum
from pyomo.core.expr.numeric_expr import LinearExpression
from egret.model_library.defn import FlowType, CoordinateType, ApproximationType
from math import tan,  radians

def declare_var_vr(model, index_set, **kwargs):
    """
    Create variable for the real component of the voltage at a bus
    """
    decl.declare_var('vr', model=model, index_set=index_set, **kwargs)


def declare_var_vj(model, index_set, **kwargs):
    """
    Create variable for the imaginary component of the voltage at a bus
    """
    decl.declare_var('vj', model=model, index_set=index_set, **kwargs)


def declare_var_vm(model, index_set, **kwargs):
    """
    Create variable for the voltage magnitude of the voltage at a bus
    """
    decl.declare_var('vm', model=model, index_set=index_set, **kwargs)


def declare_var_va(model, index_set, **kwargs):
    """
    Create variable for the phase angle of the voltage at a bus
    """
    decl.declare_var('va', model=model, index_set=index_set, **kwargs)


def declare_expr_vmsq(model, index_set, coordinate_type=CoordinateType.POLAR):
    """
    Create an expression for the voltage magnitude squared at a bus
    """
    m = model
    expr_set = decl.declare_set('_expr_vmsq', model, index_set)
    m.vmsq = pe.Expression(expr_set)

    if coordinate_type == CoordinateType.RECTANGULAR:
        for bus in expr_set:
            m.vmsq[bus] = m.vr[bus] ** 2 + m.vj[bus] ** 2
    elif coordinate_type == CoordinateType.POLAR:
        for bus in expr_set:
            m.vmsq[bus] = m.vm[bus] ** 2


def declare_var_vmsq(model, index_set, **kwargs):
    """
    Create auxiliary variable for the voltage magnitude squared at a bus
    """
    decl.declare_var('vmsq', model=model, index_set=index_set, **kwargs)


def declare_eq_vmsq(model, index_set, coordinate_type=CoordinateType.POLAR):
    """
    Create a constraint relating vmsq to the voltages
    """
    m = model
    con_set = decl.declare_set('_con_eq_vmsq', model, index_set)
    m.eq_vmsq = pe.Constraint(con_set)

    if coordinate_type == CoordinateType.POLAR:
        for bus in con_set:
            m.eq_vmsq[bus] = m.vmsq[bus] == m.vm[bus] ** 2
    elif coordinate_type == CoordinateType.RECTANGULAR:
        for bus in con_set:
            m.eq_vmsq[bus] = m.vmsq[bus] == m.vr[bus]**2 + m.vj[bus]**2
    else:
        raise ValueError('unexpected coordinate_type: {0}'.format(str(coordinate_type)))


def declare_var_ir_aggregation_at_bus(model, index_set, **kwargs):
    """
    Create a variable for the aggregated real current at a bus
    """
    decl.declare_var('ir_aggregation_at_bus', model=model, index_set=index_set, **kwargs)


def declare_var_ij_aggregation_at_bus(model, index_set, **kwargs):
    """
    Create a variable for the aggregated imaginary current at a bus
    """
    decl.declare_var('ij_aggregation_at_bus', model=model, index_set=index_set, **kwargs)


def declare_var_pl(model, index_set, **kwargs):
    """
    Create variable for the real power load at a bus
    """
    decl.declare_var('pl', model=model, index_set=index_set, **kwargs)


def declare_var_ql(model, index_set, **kwargs):
    """
    Create variable for the reactive power load at a bus
    """
    decl.declare_var('ql', model=model, index_set=index_set, **kwargs)

def declare_var_p_nw(model, index_set, **kwargs):
    """
    Create variable for the net real power withdrawals at a bus
    """
    decl.declare_var('p_nw', model=model, index_set=index_set, **kwargs)

def declare_var_q_nw(model, index_set, **kwargs):
    """
    Create variable for the net reactive power withdrawals at a bus
    """
    decl.declare_var('q_nw', model=model, index_set=index_set, **kwargs)


def declare_expr_shunt_power_at_bus(model, index_set, shunt_attrs,
                                    coordinate_type=CoordinateType.POLAR):
    """
    Create the expression for the shunt power at the bus
    """
    m = model
    expr_set = decl.declare_set('_expr_shunt_at_bus_set', model, index_set)

    m.shunt_p = pe.Expression(expr_set, initialize=0.0)
    m.shunt_q = pe.Expression(expr_set, initialize=0.0)

    if coordinate_type == CoordinateType.POLAR:
        for bus_name in expr_set:
            if bus_name in shunt_attrs['bus']:
                vmsq = m.vm[bus_name]**2
                m.shunt_p[bus_name] = shunt_attrs['gs'][bus_name]*vmsq
                m.shunt_q[bus_name] = -shunt_attrs['bs'][bus_name]*vmsq
    elif coordinate_type == CoordinateType.RECTANGULAR:
        for bus_name in expr_set:
            if bus_name in shunt_attrs['bus']:
                vmsq = m.vr[bus_name]**2 + m.vj[bus_name]**2
                m.shunt_p[bus_name] = shunt_attrs['gs'][bus_name]*vmsq
                m.shunt_q[bus_name] = -shunt_attrs['bs'][bus_name]*vmsq

def _get_dc_dicts(dc_inlet_branches_by_bus, dc_outlet_branches_by_bus, con_set):
    if dc_inlet_branches_by_bus is None:
        assert dc_outlet_branches_by_bus is None
        dc_inlet_branches_by_bus = {bn:() for bn in con_set}
    if dc_outlet_branches_by_bus is None:
        dc_outlet_branches_by_bus = dc_inlet_branches_by_bus
    return dc_inlet_branches_by_bus, dc_outlet_branches_by_bus

def declare_expr_p_net_withdraw_at_bus(model, index_set, bus_p_loads, gens_by_bus, bus_gs_fixed_shunts,
                                       dc_inlet_branches_by_bus=None, dc_outlet_branches_by_bus=None,
                                       vm_by_bus=None, **kwargs):
    """
    Create a named pyomo expression for bus net withdraw
    """
    m = model
    decl.declare_expr('p_nw', model, index_set)

    dc_inlet_branches_by_bus, dc_outlet_branches_by_bus = _get_dc_dicts(dc_inlet_branches_by_bus,
                                                                        dc_outlet_branches_by_bus,
                                                                        index_set)

    if kwargs and vm_by_bus is not None:
        for idx,val in kwargs.items():
            if idx=='linearize_shunts' and val==True:
                for b in index_set:
                    m.p_nw[b] = ( bus_gs_fixed_shunts[b] * (2 * vm_by_bus[b] * m.vm[b] - vm_by_bus[b] ** 2)
                                + (m.pl[b] if bus_p_loads[b] != 0.0 else 0.0)
                                - sum(m.pg[g] for g in gens_by_bus[b])
                                + sum(m.dcpf[branch_name] for branch_name in dc_outlet_branches_by_bus[b])
                                - sum(m.dcpf[branch_name] for branch_name in dc_inlet_branches_by_bus[b])
                                )
                return
            if idx=='linearize_shunts' and val==False:
                for b in index_set:
                    m.p_nw[b] = ( bus_gs_fixed_shunts[b] * vm_by_bus[b] ** 2
                                + (m.pl[b] if bus_p_loads[b] != 0.0 else 0.0)
                                - sum(m.pg[g] for g in gens_by_bus[b])
                                + sum(m.dcpf[branch_name] for branch_name in dc_outlet_branches_by_bus[b])
                                - sum(m.dcpf[branch_name] for branch_name in dc_inlet_branches_by_bus[b])
                                )
                return

    for b in index_set:
        m.p_nw[b] = ( bus_gs_fixed_shunts[b]
                    + ( m.pl[b] if bus_p_loads[b] != 0.0 else 0.0 )
                    - sum( m.pg[g] for g in gens_by_bus[b] )
                    + sum(m.dcpf[branch_name] for branch_name in dc_outlet_branches_by_bus[b])
                    - sum(m.dcpf[branch_name] for branch_name in dc_inlet_branches_by_bus[b])
                    )

def declare_eq_p_net_withdraw_at_bus(model, index_set, bus_p_loads, gens_by_bus, bus_gs_fixed_shunts,
                                     dc_inlet_branches_by_bus=None, dc_outlet_branches_by_bus=None,
                                     vm_by_bus=None, **kwargs):
    """
    Create a named pyomo constraint for bus net withdraw
    """
    m = model
    con_set = decl.declare_set('_con_eq_p_net_withdraw_at_bus', model, index_set)

    dc_inlet_branches_by_bus, dc_outlet_branches_by_bus = _get_dc_dicts(dc_inlet_branches_by_bus,
                                                                        dc_outlet_branches_by_bus,
                                                                        index_set)

    m.eq_p_net_withdraw_at_bus = pe.Constraint(con_set)
    constr = m.eq_p_net_withdraw_at_bus

    if kwargs and vm_by_bus is not None:
        for idx,val in kwargs.items():
            if idx=='linearize_shunts' and val==True:
                for b in index_set:
                    constr[b] = m.p_nw[b] == ( bus_gs_fixed_shunts[b] * (2 * vm_by_bus[b] * m.vm[b] - vm_by_bus[b] ** 2)
                                + (m.pl[b] if bus_p_loads[b] != 0.0 else 0.0)
                                - sum(m.pg[g] for g in gens_by_bus[b])
                                + sum(m.dcpf[branch_name] for branch_name in dc_outlet_branches_by_bus[b])
                                - sum(m.dcpf[branch_name] for branch_name in dc_inlet_branches_by_bus[b])
                                )
                return
            if idx=='linearize_shunts' and val==False:
                for b in index_set:
                    constr[b] = m.p_nw[b] == ( bus_gs_fixed_shunts[b] * vm_by_bus[b] ** 2
                                + (m.pl[b] if bus_p_loads[b] != 0.0 else 0.0)
                                - sum(m.pg[g] for g in gens_by_bus[b])
                                + sum(m.dcpf[branch_name] for branch_name in dc_outlet_branches_by_bus[b])
                                - sum(m.dcpf[branch_name] for branch_name in dc_inlet_branches_by_bus[b])
                                )
                return
    else:
        for b in index_set:
            constr[b] = m.p_nw[b] == ( bus_gs_fixed_shunts[b]
                        + ( m.pl[b] if bus_p_loads[b] != 0.0 else 0.0 )
                        - sum( m.pg[g] for g in gens_by_bus[b] )
                        + sum(m.dcpf[branch_name] for branch_name in dc_outlet_branches_by_bus[b])
                        - sum(m.dcpf[branch_name] for branch_name in dc_inlet_branches_by_bus[b])
                        )


def declare_expr_q_net_withdraw_at_bus(model, index_set, bus_q_loads, gens_by_bus, bus_bs_fixed_shunts,
                                       vm_by_bus=None, **kwargs):
    """
    Create a named pyomo expression for bus net withdraw
    """
    m = model
    decl.declare_expr('q_nw', model, index_set)

    if kwargs and vm_by_bus is not None:
        for idx,val in kwargs.items():
            if idx=='linearize_shunts' and val==True:
                for b in index_set:
                    m.q_nw[b] = (-bus_bs_fixed_shunts[b] * (2 * vm_by_bus[b] * m.vm[b] - vm_by_bus[b] ** 2)
                                + (m.ql[b] if bus_q_loads[b] != 0.0 else 0.0)
                                - sum(m.qg[g] for g in gens_by_bus[b])
                                 )
                return
            if idx=='linearize_shunts' and val==False:
                for b in index_set:
                    m.q_nw[b] = (-bus_bs_fixed_shunts[b] * vm_by_bus[b] ** 2
                                + (m.ql[b] if bus_q_loads[b] != 0.0 else 0.0)
                                - sum(m.qg[g] for g in gens_by_bus[b])
                                )
                return

    for b in index_set:
        m.q_nw[b] = (-bus_bs_fixed_shunts[b]
                    + ( m.ql[b] if bus_q_loads[b] != 0.0 else 0.0 )
                    - sum( m.qg[g] for g in gens_by_bus[b] )
                    )

def declare_eq_q_net_withdraw_at_bus(model, index_set, bus_q_loads, gens_by_bus, bus_bs_fixed_shunts,
                                     vm_by_bus=None, **kwargs):
    """
    Create a named pyomo constraint for bus net withdraw
    """
    m = model
    con_set = decl.declare_set('_con_eq_q_net_withdraw_at_bus', model, index_set)

    m.eq_q_net_withdraw_at_bus = pe.Constraint(con_set)
    constr = m.eq_q_net_withdraw_at_bus

    if kwargs and vm_by_bus is not None:
        for idx,val in kwargs.items():
            if idx=='linearize_shunts' and val==True:
                for b in index_set:
                    constr[b] = m.q_nw[b] == (-bus_bs_fixed_shunts[b] * (2 * vm_by_bus[b] * m.vm[b] - vm_by_bus[b] ** 2)
                                + (m.ql[b] if bus_q_loads[b] != 0.0 else 0.0)
                                - sum(m.qg[g] for g in gens_by_bus[b])
                                 )
                return
            if idx=='linearize_shunts' and val==False:
                for b in index_set:
                    constr[b] = m.q_nw[b] == (-bus_bs_fixed_shunts[b] * vm_by_bus[b] ** 2
                                + (m.ql[b] if bus_q_loads[b] != 0.0 else 0.0)
                                - sum(m.qg[g] for g in gens_by_bus[b])
                                )
                return

    for b in index_set:
        constr[b] = m.q_nw[b] == (-bus_bs_fixed_shunts[b]
                    + ( m.ql[b] if bus_q_loads[b] != 0.0 else 0.0 )
                    - sum( m.qg[g] for g in gens_by_bus[b] )
                    )

def declare_eq_ref_bus_nonzero(model, ref_angle, ref_bus):
    """
    Create an equality constraint to enforce tan(\theta) = vj/vr at  the reference bus
    """
    m = model
    m.eq_ref_bus_nonzero = pe.Constraint(expr = tan(radians(ref_angle)) * m.vr[ref_bus] == m.vj[ref_bus])

def declare_eq_i_aggregation_at_bus(model, index_set,
                                    bus_bs_fixed_shunts, bus_gs_fixed_shunts,
                                    inlet_branches_by_bus, outlet_branches_by_bus):
    """
    Create the equality constraints for the aggregated real and imaginary
    currents at the bus
    """
    m = model
    con_set = decl.declare_set('_con_eq_i_aggregation_at_bus_set', model, index_set)

    m.eq_ir_aggregation_at_bus = pe.Constraint(con_set)
    m.eq_ij_aggregation_at_bus = pe.Constraint(con_set)

    for bus_name in con_set:
        ir_expr = sum([m.ifr[branch_name] for branch_name in outlet_branches_by_bus[bus_name]])
        ir_expr += sum([m.itr[branch_name] for branch_name in inlet_branches_by_bus[bus_name]])
        ij_expr = sum([m.ifj[branch_name] for branch_name in outlet_branches_by_bus[bus_name]])
        ij_expr += sum([m.itj[branch_name] for branch_name in inlet_branches_by_bus[bus_name]])

        if bus_bs_fixed_shunts[bus_name] != 0.0:
            ir_expr -= bus_bs_fixed_shunts[bus_name] * m.vj[bus_name]
            ij_expr += bus_bs_fixed_shunts[bus_name] * m.vr[bus_name]
        if bus_gs_fixed_shunts[bus_name] != 0.0:
            ir_expr += bus_gs_fixed_shunts[bus_name] * m.vr[bus_name]
            ij_expr += bus_gs_fixed_shunts[bus_name] * m.vj[bus_name]

        ir_expr -= m.ir_aggregation_at_bus[bus_name]
        ij_expr -= m.ij_aggregation_at_bus[bus_name]

        m.eq_ir_aggregation_at_bus[bus_name] = ir_expr == 0
        m.eq_ij_aggregation_at_bus[bus_name] = ij_expr == 0


def declare_eq_p_balance_ed(model, index_set, bus_p_loads, gens_by_bus, bus_gs_fixed_shunts, **rhs_kwargs):
    """
    Create the equality constraints for the system-wide real power balance.

    NOTE: Equation build orientates constants to the RHS in order to compute the correct dual variable sign
    """
    m = model

    p_expr = sum(m.pg[gen_name] for bus_name in index_set for gen_name in gens_by_bus[bus_name])
    p_expr -= sum(m.pl[bus_name] for bus_name in index_set if bus_p_loads[bus_name] is not None)
    p_expr -= sum(bus_gs_fixed_shunts[bus_name] for bus_name in index_set if bus_gs_fixed_shunts[bus_name] != 0.0)

    relaxed_balance = False

    if rhs_kwargs:
        for idx, val in rhs_kwargs.items():
            if idx == 'include_feasibility_load_shed':
                p_expr += eval("m." + val)
            if idx == 'include_feasibility_over_generation':
                p_expr -= eval("m." + val)
            if idx == 'include_losses':
                p_expr -= sum(m.pfl[branch_name] for branch_name in val)
            if idx == 'relax_balance':
                relaxed_balance = True

    if relaxed_balance:
        m.eq_p_balance = pe.Constraint(expr=p_expr >= 0.0)
    else:
        m.eq_p_balance = pe.Constraint(expr=p_expr == 0.0)


def declare_eq_p_balance_lopf(model, index_set, bus_p_loads, gens_by_bus, bus_gs_fixed_shunts, vm_by_bus, **rhs_kwargs):
    """
    Create the equality constraints for the system-wide real power balance.

    NOTE: Equation build orientates constants to the RHS in order to compute the correct dual variable sign
    """
    m = model

    p_expr = sum(m.pg[gen_name] for bus_name in index_set for gen_name in gens_by_bus[bus_name])
    p_expr -= sum(m.pl[bus_name] for bus_name in index_set if bus_p_loads[bus_name] is not None)

    relaxed_balance = False

    if rhs_kwargs:
        for idx,val in rhs_kwargs.items():
            if idx == 'include_feasibility_load_shed':
                p_expr += eval("m." + val)
            if idx == 'include_feasibility_over_generation':
                p_expr -= eval("m." + val)
            if idx == 'include_branch_losses':
                pass                            # branch losses are added to the constraint after updating pfl constraints
            if idx == 'include_system_losses':
                p_expr -= m.ploss
            if idx == 'relax_balance':
                relaxed_balance = True
            if idx == 'linearize_shunts':
                if val == True:
                    p_expr -= sum( bus_gs_fixed_shunts[b] * (2 * vm_by_bus[b] * m.vm[b] - vm_by_bus[b] ** 2) \
                        for b in index_set if bus_gs_fixed_shunts[b] != 0.0)
                elif val == False:
                    p_expr -= sum( bus_gs_fixed_shunts[b] * vm_by_bus[b] ** 2 \
                        for b in index_set if bus_gs_fixed_shunts[b] != 0.0)
                else:
                    raise Exception('linearize_shunts option is invalid.')

    if relaxed_balance:
        m.eq_p_balance = pe.Constraint(expr = p_expr >= 0.0)
    else:
        m.eq_p_balance = pe.Constraint(expr = p_expr == 0.0)

def declare_eq_q_balance_lopf(model, index_set, bus_q_loads, gens_by_bus, bus_bs_fixed_shunts, vm_by_bus, **rhs_kwargs):
    """
    Create the equality constraints for the system-wide real power balance.

    NOTE: Equation build orientates constants to the RHS in order to compute the correct dual variable sign
    """
    m = model

    q_expr = sum(m.qg[gen_name] for bus_name in index_set for gen_name in gens_by_bus[bus_name])
    q_expr -= sum(m.ql[bus_name] for bus_name in index_set if bus_q_loads[bus_name] is not None)

    relaxed_balance = False

    if rhs_kwargs:
        for idx,val in rhs_kwargs.items():
            if idx == 'include_reactive_load_shed':
                q_expr += eval("m." + val)
            if idx == 'include_reactive_over_generation':
                q_expr -= eval("m." + val)
            if idx == 'include_branch_losses':
                pass                            # branch losses are added to the constraint after updating qfl constraints
            if idx == 'include_system_losses':
                q_expr -= m.qloss
            if idx == 'relax_balance':
                relaxed_balance = True
            if idx == 'linearize_shunts':
                if val == True:
                    q_expr -= sum( bus_bs_fixed_shunts[b] * (2 * vm_by_bus[b] * m.vm[b] - vm_by_bus[b] ** 2) \
                        for b in index_set if bus_bs_fixed_shunts[b] != 0.0)
                elif val == False:
                    q_expr -= sum( bus_bs_fixed_shunts[b] * vm_by_bus[b] ** 2 \
                        for b in index_set if bus_bs_fixed_shunts[b] != 0.0)
                else:
                    raise Exception('linearize_shunts option is invalid.')

    if relaxed_balance:
        m.eq_q_balance = pe.Constraint(expr = q_expr >= 0.0)
    else:
        m.eq_q_balance = pe.Constraint(expr = q_expr == 0.0)

def declare_eq_ploss_sum_of_pfl(model, index_set):
    """
    Create the equality constraint or expression for total real power losses (from PTDF approximation)
    """
    m=model

    ploss_is_var = isinstance(m.ploss, pe.Var)
    if ploss_is_var:
        m.eq_ploss = pe.Constraint()
    else:
        if not isinstance(m.ploss, pe.Expression):
            raise Exception("Unrecognized type for m.ploss", m.ploss.pprint())

    expr = sum(m.pfl[bn] for bn in index_set)

    if ploss_is_var:
        m.eq_ploss = m.ploss == expr
    else:
        m.ploss = expr

def declare_eq_qloss_sum_of_qfl(model, index_set):
    """
    Create the equality constraint or expression for total real power losses (from PTDF approximation)
    """
    m=model

    qloss_is_var = isinstance(m.qloss, pe.Var)
    if qloss_is_var:
        m.eq_qloss = pe.Constraint()
    else:
        if not isinstance(m.qloss, pe.Expression):
            raise Exception("Unrecognized type for m.qloss", m.qloss.pprint())

    expr = sum(m.qfl[bn] for bn in index_set)

    if qloss_is_var:
        m.eq_qloss = m.qloss == expr
    else:
        m.qloss = expr

def declare_eq_ploss_ptdf_approx(model, PTDF, rel_ptdf_tol=None, abs_ptdf_tol=None, use_residuals=False):
    """
    Create the equality constraint or expression for total real power losses (from PTDF approximation)
    """

    m = model

    ploss_is_var = isinstance(m.ploss, pe.Var)
    if ploss_is_var:
        m.eq_ploss = pe.Constraint()
    else:
        if not isinstance(m.ploss, pe.Expression):
            raise Exception("Unrecognized type for m.ploss", m.ploss.pprint())

    if rel_ptdf_tol is None:
        rel_ptdf_tol = 0.
    if abs_ptdf_tol is None:
        abs_ptdf_tol = 0.

    expr = get_ploss_expr_ptdf_approx(m, PTDF, abs_ptdf_tol=abs_ptdf_tol, rel_ptdf_tol=rel_ptdf_tol, use_residuals=use_residuals)

    if ploss_is_var:
        m.eq_ploss = m.ploss == expr
    else:
        m.ploss = expr

def get_ploss_expr_ptdf_approx(m, PTDF, abs_ptdf_tol=None, rel_ptdf_tol=None, use_residuals=False):

    if not use_residuals:
        const = PTDF.get_lossoffset()
        iterator = PTDF.get_lossfactor_iterator()
    else:
        const = PTDF.get_lossoffset_resid()
        iterator = PTDF.get_lossfactor_resid_iterator()
    max_coef = PTDF.get_lossfactor_abs_max()
    ptdf_tol = max(abs_ptdf_tol, rel_ptdf_tol*max_coef)
    m_p_nw = m.p_nw
    ## if model.p_nw is Var, we can use LinearExpression
    ## to build these dense constraints much faster
    coef_list = []
    var_list = []
    for bus_name, coef in iterator:
        if abs(coef) >= ptdf_tol:
            coef_list.append(coef)
            var_list.append(m_p_nw[bus_name])

    if use_residuals:
        for i in m._idx_monitored:
            bn = PTDF.branches_keys_masked[i]
            coef_list.append(1)
            var_list.append(m.pfl[bn])

    if isinstance(m_p_nw, pe.Var):
        expr = LinearExpression(linear_vars=var_list, linear_coefs=coef_list, constant=const)
    else:
        expr = quicksum( (coef*var for coef, var in zip(coef_list, var_list)), start=const, linear=True)

    return expr

def declare_eq_qloss_ptdf_approx(model, PTDF, rel_ptdf_tol=None, abs_ptdf_tol=None, use_residuals=False):
    """
    Create the equality constraint or expression for total real power losses (from PTDF approximation)
    """

    m = model

    qloss_is_var = isinstance(m.qloss, pe.Var)
    if qloss_is_var:
        m.eq_qloss = pe.Constraint()
    else:
        if not isinstance(m.qloss, pe.Expression):
            raise Exception("Unrecognized type for m.qloss", m.qloss.pprint())

    if rel_ptdf_tol is None:
        rel_ptdf_tol = 0.
    if abs_ptdf_tol is None:
        abs_ptdf_tol = 0.

    expr = get_qloss_expr_ptdf_approx(m, PTDF, abs_ptdf_tol=abs_ptdf_tol, rel_ptdf_tol=rel_ptdf_tol, use_residuals=use_residuals)

    if qloss_is_var:
        m.eq_qloss = m.qloss == expr
    else:
        m.qloss = expr

def get_qloss_expr_ptdf_approx(m, PTDF, abs_ptdf_tol=None, rel_ptdf_tol=None, use_residuals=False):

    if not use_residuals:
        const = PTDF.get_qlossoffset()
        iterator = PTDF.get_qlossfactor_iterator()
    else:
        const = PTDF.get_qlossoffset_resid()
        iterator = PTDF.get_qlossfactor_resid_iterator()
    max_coef = PTDF.get_qlossfactor_abs_max()
    ptdf_tol = max(abs_ptdf_tol, rel_ptdf_tol*max_coef)
    m_q_nw = m.q_nw
    ## if model.q_nw is Var, we can use LinearExpression
    ## to build these dense constraints much faster
    coef_list = []
    var_list = []
    for bus_name, coef in iterator:
        if abs(coef) >= ptdf_tol:
            coef_list.append(coef)
            var_list.append(m_q_nw[bus_name])

    if use_residuals:
        for i in m._idx_monitored:
            bn = PTDF.branches_keys[i]
            coef_list.append(1)
            var_list.append(m.qfl[bn])

    if isinstance(m_q_nw, pe.Var):
        expr = LinearExpression(linear_vars=var_list, linear_coefs=coef_list, constant=const)
    else:
        expr = quicksum( (coef*var for coef, var in zip(coef_list, var_list)), start=const, linear=True)

    return expr

def declare_eq_bus_vm_approx(model, index_set, PTDF=None, rel_ptdf_tol=None, abs_ptdf_tol=None):
    """
    Create the equality constraints or expressions for voltage magnitude (from PTDF
    approximation) at the bus
    """

    m = model

    con_set = decl.declare_set("_con_eq_bus_vm_approx_set", model, index_set)

    vm_is_var = isinstance(m.vm, pe.Var)

    if vm_is_var:
        m.eq_vm_bus = pe.Constraint(con_set)
    else:
        if not isinstance(m.vm, pe.Expression):
            raise Exception("Unrecognized type for m.vm", m.vm.pprint())

    if PTDF is None:
        return

    for bus_name in con_set:
        expr = \
            get_vm_expr_ptdf_approx(m, bus_name, PTDF, rel_ptdf_tol=rel_ptdf_tol, abs_ptdf_tol=abs_ptdf_tol)

        if vm_is_var:
            m.eq_vm_bus[bus_name] = \
                m.vm[bus_name] == expr
        else:
            m.vm[bus_name] = expr

def get_vm_expr_ptdf_approx(model, bus_name, PTDF, rel_ptdf_tol=None, abs_ptdf_tol=None):
    """
    Create a pyomo reactive power flow expression from PTDF matrix
    """
    if rel_ptdf_tol is None:
        rel_ptdf_tol = 0.
    if abs_ptdf_tol is None:
        abs_ptdf_tol = 0.

    const = PTDF.get_bus_vdf_const(bus_name)

    max_coef = PTDF.get_bus_vdf_abs_max(bus_name)

    ptdf_tol = max(abs_ptdf_tol, rel_ptdf_tol*max_coef)
    ## NOTE: It would be easy to hold on to the 'ptdf' dictionary here, if we wanted to
    m_q_nw = model.q_nw
    qnw_is_var = isinstance(m_q_nw, pe.Var)
    ## if model.q_nw is Var, we can use LinearExpression
    ## to build these dense constraints much faster
    coef_list = []
    var_list = []
    for bn, coef in PTDF.get_bus_vdf_iterator(bus_name):
        if abs(coef) >= ptdf_tol:
            coef_list.append(coef)
            var_list.append(m_q_nw[bn])
        elif qnw_is_var:
            const += coef * m_q_nw[bn].value
        else:
            const += coef * m_q_nw[bn].expr()

    if qnw_is_var:
        expr = LinearExpression(linear_vars=var_list, linear_coefs=coef_list, constant=const)
    else:
        expr = quicksum( (coef*var for coef, var in zip(coef_list, var_list)), start=const, linear=True)

    return expr

def declare_eq_p_balance_dc_approx(model, index_set,
                                   bus_p_loads,
                                   gens_by_bus,
                                   bus_gs_fixed_shunts,
                                   inlet_branches_by_bus, outlet_branches_by_bus,
                                   approximation_type=ApproximationType.BTHETA,
                                   dc_inlet_branches_by_bus=None,
                                   dc_outlet_branches_by_bus=None,
                                   **rhs_kwargs):
    """
    Create the equality constraints for the real power balance
    at a bus using the variables for real power flows, respectively.

    NOTE: Equation build orientates constants to the RHS in order to compute the correct dual variable sign
    """
    m = model
    con_set = decl.declare_set('_con_eq_p_balance', model, index_set)

    m.eq_p_balance = pe.Constraint(con_set)

    for bus_name in con_set:
        if approximation_type == ApproximationType.BTHETA:
            p_expr = -sum(m.pf[branch_name] for branch_name in outlet_branches_by_bus[bus_name])
            p_expr += sum(m.pf[branch_name] for branch_name in inlet_branches_by_bus[bus_name])
        elif approximation_type == ApproximationType.BTHETA_LOSSES:
            p_expr = -0.5*sum(m.pfl[branch_name] for branch_name in inlet_branches_by_bus[bus_name])
            p_expr -= 0.5*sum(m.pfl[branch_name] for branch_name in outlet_branches_by_bus[bus_name])
            p_expr -= sum(m.pf[branch_name] for branch_name in outlet_branches_by_bus[bus_name])
            p_expr += sum(m.pf[branch_name] for branch_name in inlet_branches_by_bus[bus_name])

        if dc_inlet_branches_by_bus is not None:
            p_expr -= sum(m.dcpf[branch_name] for branch_name in dc_outlet_branches_by_bus[bus_name])
            p_expr += sum(m.dcpf[branch_name] for branch_name in dc_inlet_branches_by_bus[bus_name])

        if bus_gs_fixed_shunts[bus_name] != 0.0:
            p_expr -= bus_gs_fixed_shunts[bus_name]

        if bus_p_loads[bus_name] != 0.0: # only applies to fixed loads, otherwise may cause an error
            p_expr -= m.pl[bus_name]

        if rhs_kwargs:
            k = bus_name
            for idx, val in rhs_kwargs.items():
                if isinstance(val, tuple):
                    val,key = val
                    k = (key,bus_name)
                if not k in eval("m." + val).index_set():
                    continue
                if idx == 'include_feasibility_load_shed':
                    p_expr += eval("m." + val)[k]
                if idx == 'include_feasibility_over_generation':
                    p_expr -= eval("m." + val)[k]

        for gen_name in gens_by_bus[bus_name]:
            p_expr += m.pg[gen_name]

        m.eq_p_balance[bus_name] = \
            p_expr == 0.0


def declare_eq_p_balance(model, index_set,
                         bus_p_loads,
                         gens_by_bus,
                         bus_gs_fixed_shunts,
                         inlet_branches_by_bus, outlet_branches_by_bus,
                         **rhs_kwargs):
    """
    Create the equality constraints for the real power balance
    at a bus using the variables for real power flows, respectively.

    NOTE: Equation build orientates constants to the RHS in order to compute the correct dual variable sign
    """

    m = model
    con_set = decl.declare_set('_con_eq_p_balance', model, index_set)

    m.eq_p_balance = pe.Constraint(con_set)

    for bus_name in con_set:
        p_expr = -sum([m.pf[branch_name] for branch_name in outlet_branches_by_bus[bus_name]])
        p_expr -= sum([m.pt[branch_name] for branch_name in inlet_branches_by_bus[bus_name]])

        if bus_gs_fixed_shunts[bus_name] != 0.0:
            vmsq = m.vmsq[bus_name]
            p_expr -= bus_gs_fixed_shunts[bus_name] * vmsq

        if bus_p_loads[bus_name] != 0.0: # only applies to fixed loads, otherwise may cause an error
            p_expr -= m.pl[bus_name]

        if rhs_kwargs:
            for idx, val in rhs_kwargs.items():
                if idx == 'include_feasibility_load_shed':
                    p_expr += eval("m." + val)[bus_name]
                if idx == 'include_feasibility_over_generation':
                    p_expr -= eval("m." + val)[bus_name]

        for gen_name in gens_by_bus[bus_name]:
            p_expr += m.pg[gen_name]

        m.eq_p_balance[bus_name] = \
            p_expr == 0.0


def declare_eq_p_balance_with_i_aggregation(model, index_set,
                                            bus_p_loads,
                                            gens_by_bus,
                                            **rhs_kwargs):
    """
    Create the equality constraints for the real power balance
    at a bus using the variables for real power flows, respectively.

    NOTE: Equation build orientates constants to the RHS in order to compute the correct dual variable sign
    """
    m = model
    con_set = decl.declare_set('_con_eq_p_balance', model, index_set)

    m.eq_p_balance = pe.Constraint(con_set)

    for bus_name in con_set:
        p_expr = -m.vr[bus_name] * m.ir_aggregation_at_bus[bus_name] + \
                 -m.vj[bus_name] * m.ij_aggregation_at_bus[bus_name]

        if bus_p_loads[bus_name] != 0.0: # only applies to fixed loads, otherwise may cause an error
            p_expr -= m.pl[bus_name]

        if rhs_kwargs:
            for idx, val in rhs_kwargs.items():
                if idx == 'include_feasibility_load_shed':
                    p_expr += eval("m." + val)[bus_name]
                if idx == 'include_feasibility_over_generation':
                    p_expr -= eval("m." + val)[bus_name]

        for gen_name in gens_by_bus[bus_name]:
            p_expr += m.pg[gen_name]

        m.eq_p_balance[bus_name] = \
            p_expr == 0.0


def declare_eq_q_balance(model, index_set,
                         bus_q_loads,
                         gens_by_bus,
                         bus_bs_fixed_shunts,
                         inlet_branches_by_bus, outlet_branches_by_bus,
                         **rhs_kwargs):
    """
    Create the equality constraints for the reactive power balance
    at a bus using the variables for reactive power flows, respectively.

    NOTE: Equation build orientates constants to the RHS in order to compute the correct dual variable sign
    """
    m = model
    con_set = decl.declare_set('_con_eq_q_balance', model, index_set)

    m.eq_q_balance = pe.Constraint(con_set)

    for bus_name in con_set:
        q_expr = -sum([m.qf[branch_name] for branch_name in outlet_branches_by_bus[bus_name]])
        q_expr -= sum([m.qt[branch_name] for branch_name in inlet_branches_by_bus[bus_name]])

        if bus_bs_fixed_shunts[bus_name] != 0.0:
            vmsq = m.vmsq[bus_name]
            q_expr += bus_bs_fixed_shunts[bus_name] * vmsq

        if bus_q_loads[bus_name] != 0.0: # only applies to fixed loads, otherwise may cause an error
            q_expr -= m.ql[bus_name]

        if rhs_kwargs:
            for idx, val in rhs_kwargs.items():
                if idx == 'include_feasibility_load_shed':
                    q_expr += eval("m." + val)[bus_name]
                if idx == 'include_feasibility_over_generation':
                    q_expr -= eval("m." + val)[bus_name]

        for gen_name in gens_by_bus[bus_name]:
            q_expr += m.qg[gen_name]

        m.eq_q_balance[bus_name] = \
            q_expr == 0.0


def declare_eq_q_balance_with_i_aggregation(model, index_set,
                                            bus_q_loads,
                                            gens_by_bus,
                                            **rhs_kwargs):
    """
    Create the equality constraints for the reactive power balance
    at a bus using the variables for reactive power flows, respectively.

    NOTE: Equation build orientates constants to the RHS in order to compute the correct dual variable sign
    """
    m = model
    con_set = decl.declare_set('_con_eq_q_balance', model, index_set)

    m.eq_q_balance = pe.Constraint(con_set)

    for bus_name in con_set:
        q_expr = m.vr[bus_name] * m.ij_aggregation_at_bus[bus_name] + \
                 -m.vj[bus_name] * m.ir_aggregation_at_bus[bus_name]

        if bus_q_loads[bus_name] != 0.0: # only applies to fixed loads, otherwise may cause an error
            q_expr -= m.ql[bus_name]

        if rhs_kwargs:
            for idx, val in rhs_kwargs.items():
                if idx == 'include_feasibility_load_shed':
                    q_expr += eval("m." + val)[bus_name]
                if idx == 'include_feasibility_over_generation':
                    q_expr -= eval("m." + val)[bus_name]

        for gen_name in gens_by_bus[bus_name]:
            q_expr += m.qg[gen_name]

        m.eq_q_balance[bus_name] = \
            q_expr == 0.0


def declare_ineq_vm_bus_lbub(model, index_set, buses, coordinate_type=CoordinateType.POLAR):
    """
    Create the inequalities for the voltage magnitudes from the
    voltage variables
    """
    m = model
    con_set = decl.declare_set('_con_ineq_vm_bus_lbub',
                               model=model, index_set=index_set)

    m.ineq_vm_bus_lb = pe.Constraint(con_set)
    m.ineq_vm_bus_ub = pe.Constraint(con_set)

    if coordinate_type == CoordinateType.POLAR:
        for bus_name in con_set:
            m.ineq_vm_bus_lb[bus_name] = \
                buses[bus_name]['v_min'] <= m.vm[bus_name]
            m.ineq_vm_bus_ub[bus_name] = \
                m.vm[bus_name] <= buses[bus_name]['v_max']
    elif coordinate_type == CoordinateType.RECTANGULAR:
        for bus_name in con_set:
            m.ineq_vm_bus_lb[bus_name] = \
                buses[bus_name]['v_min']**2 <= m.vr[bus_name]**2 + m.vj[bus_name]**2
            m.ineq_vm_bus_ub[bus_name] = \
                m.vr[bus_name]**2 + m.vj[bus_name]**2 <= buses[bus_name]['v_max']**2
