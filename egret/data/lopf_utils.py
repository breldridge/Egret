#  ___________________________________________________________________________
#
#  EGRET: Electrical Grid Research and Engineering Tools
#  Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
This module contains some helpers and data for test_approximations.py
"""
import os

case_names = ['pglib_opf_case3_lmbd',
              'pglib_opf_case5_pjm',
              'pglib_opf_case14_ieee',
              'pglib_opf_case24_ieee_rts',
              'pglib_opf_case30_as',
              'pglib_opf_case30_fsr',
              'pglib_opf_case30_ieee',
              'pglib_opf_case39_epri',
              'pglib_opf_case57_ieee',
              'pglib_opf_case73_ieee_rts',
              'pglib_opf_case89_pegase',
              'pglib_opf_case118_ieee',
              'pglib_opf_case162_ieee_dtc',
              'pglib_opf_case179_goc',
              'pglib_opf_case200_tamu',
              'pglib_opf_case240_pserc',
              'pglib_opf_case300_ieee',
              'pglib_opf_case500_tamu',
              'pglib_opf_case588_sdet',
              'pglib_opf_case1354_pegase',
              'pglib_opf_case1888_rte',
              'pglib_opf_case1951_rte',
              'pglib_opf_case2000_tamu',
              'pglib_opf_case2316_sdet',
              'pglib_opf_case2383wp_k',
              'pglib_opf_case2736sp_k',
              'pglib_opf_case2737sop_k',
              'pglib_opf_case2746wop_k',
              'pglib_opf_case2746wp_k',
              'pglib_opf_case2848_rte',
              'pglib_opf_case2853_sdet',
              'pglib_opf_case2868_rte',
              'pglib_opf_case2869_pegase',
              'pglib_opf_case3012wp_k',
              'pglib_opf_case3120sp_k',
              'pglib_opf_case3375wp_k',
              'pglib_opf_case4661_sdet',
              'pglib_opf_case6468_rte',
              'pglib_opf_case6470_rte',
              'pglib_opf_case6495_rte',
              'pglib_opf_case6515_rte',
              'pglib_opf_case9241_pegase',
              'pglib_opf_case10000_tamu',
              'pglib_opf_case13659_pegase',
              ]
idx_deca = case_names.index('pglib_opf_case118_ieee')
idx_kilo = case_names.index('pglib_opf_case1354_pegase')
cases_0toC = case_names[0:idx_deca]
cases_CtoM = case_names[idx_deca:idx_kilo]
cases_MtoX = case_names[idx_kilo:-1]

test_cases = [os.path.join('../../../download/pglib-opf-master/', f + '.m') for f in case_names]

def idx_to_test_case(s):
    try:
        idx = int(s)
        tc = test_cases[idx]
        return tc
    except IndexError:
        raise SyntaxError("Index out of range of test_cases.")
    except ValueError:
        try:
            idx = case_names.index(s)
            tc = test_cases[idx]
            return tc
        except ValueError:
            raise SyntaxError(
                "Expecting argument of either A, B, C, D, E, or an index or case name from the test_cases list.")

def get_solution_file_location(test_case):
    _, case = os.path.split(test_case)
    case, _ = os.path.splitext(case)
    current_dir, current_file = os.path.split(os.path.realpath(__file__))
    solution_location = os.path.join(current_dir, 'transmission_test_instances', 'approximation_solution_files', case)

    return solution_location

def get_summary_file_location(folder):
    current_dir, current_file = os.path.split(os.path.realpath(__file__))
    location = os.path.join(current_dir, 'transmission_test_instances','approximation_summary_files', folder)

    if not os.path.exists(location):
        os.makedirs(location)

    return location

def get_sensitivity_dict(test_model_list):

    dlist = [tm for tm in test_model_list if 'dlopf' in tm]
    clist = [tm for tm in test_model_list if 'clopf' in tm]
    plist = [tm for tm in test_model_list if 'plopf' in tm]
    dense_keepers = [dlist[0], clist[0], plist[0]]

    tm_dict = {}
    for key in test_model_list:
        if 'lopf' in key:
            if key in dense_keepers:
                tm_dict[key] = True
            elif 'slopf' in key:
                tm_dict[key] = True
            else:
                tm_dict[key] = False
        elif 'acopf' in key:
            tm_dict[key] = True
        else:
            tm_dict[key] = False

    return tm_dict

def get_pareto_dict(test_model_list):

    tm_dict = {}
    for key in test_model_list:
        if 'acopf' in key or '_lazy' in key:
            tm_dict[key] = False
        elif 'qcopf' in key:
            tm_dict[key] = False
        else:
            tm_dict[key] = True

    return tm_dict

def get_case_size_dict(test_model_list):

    tm_dict = {}
    for key in test_model_list:
        if 'qcopf' in key:
            tm_dict[key] = False
        else:
            tm_dict[key] = True

    return tm_dict

def get_scatter_full_dict(test_model_list):

    tm_dict = {}
    for key in test_model_list:
        if 'acopf' in key:
            tm_dict[key] = True
        else:
            tm_dict[key] = True

    return tm_dict

def get_scatter_settings_dict(test_model_list, model='dlopf'):

    tm_dict = {}
    for key in test_model_list:
        if model in key:
            tm_dict[key] = True
        else:
            tm_dict[key] = False

    return tm_dict

def get_scatter_filtered_dict(test_model_list):

    tm_dict = {}
    for key in test_model_list:
        if 'acopf' in key:
            tm_dict[key] = True
        elif 'lopf' in key and '_e2' in key:
            tm_dict[key] = True
        else:
            tm_dict[key] = False

    return tm_dict

def get_scatter_pareto_dict(test_model_list):

    tm_dict = {}

    for key in test_model_list:
        if any(m in key for m in ['slopf','btheta']):
            tm_dict[key] = True
        elif '_e2' in key:
            tm_dict[key] = True
        else:
            tm_dict[key] = False

    return tm_dict


def get_lazy_speedup_dict(test_model_list):

    tm_dict = {}
    for key in test_model_list:
        if '_default' in key or '_lazy' in key:
            tm_dict[key] = True
        else:
            tm_dict[key] = False

    return tm_dict

def get_trunc_speedup_dict(test_model_list):

    tm_dict = {}
    for key in test_model_list:
        if 'dlopf_default' in key \
                or 'dlopf_e' in key \
                or 'clopf_default' in key \
                or 'clopf_e' in key:
            tm_dict[key] = True
        else:
            tm_dict[key] = False

    return tm_dict

def get_violation_dict(test_model_list):

    dlist = [tm for tm in test_model_list if 'dlopf' in tm]
    clist = [tm for tm in test_model_list if 'clopf' in tm]
    plist = [tm for tm in test_model_list if 'plopf' in tm]
    qlist = [tm for tm in test_model_list if 'ptdf' in tm]
    dense_keepers = []
    if bool(dlist):
        dense_keepers.append(dlist[0])
    if bool(clist):
        dense_keepers.append(clist[0])
    if bool(plist):
        dense_keepers.append(plist[0])
    if bool(qlist):
        dense_keepers.append(qlist[0])

    tm_dict = {}
    for key in test_model_list:
        if 'lopf' in key :
            tm_dict[key] = True
        else:
            tm_dict[key] = False

    return tm_dict

def get_violin_dict(test_model_list):
    tm_dict = get_case_size_dict(test_model_list)
    return tm_dict

def get_vanilla_dict(test_model_list, case=None):

    tm_dict = {}
    for key in test_model_list:
        if 'acopf' in key:
            tm_dict[key] = False
        elif 'lazy' in key:
            tm_dict[key] = False
        elif 'e2' in key or 'e4' in key:
            tm_dict[key] = False
        else:
            tm_dict[key] = True

    if case is not None:
        replace_unsolved_case(tm_dict, case)

    return tm_dict

def get_barplot_dict(test_model_list, case=None):

    tm_dict = {}
    for key in test_model_list:
        if 'lazy' in key:
            tm_dict[key] = True
        elif any([m in key for m in ['slopf','btheta']]):
            tm_dict[key] = True
        else:
            tm_dict[key] = False

    if case is not None:
        replace_unsolved_case(tm_dict, case)

    return tm_dict

def get_lazy_dict(test_model_list, case=None):

    tm_dict = {}
    for key in test_model_list:
        if any([m in key for m in ['acopf','ptdf','btheta']]):
            tm_dict[key] = False
        elif 'e2' in key or 'e4' in key:
            tm_dict[key] = False
        elif 'lazy' in key:
            tm_dict[key] = True
        else:
            tm_dict[key] = True

    if case is not None:
        replace_unsolved_case(tm_dict, case)

    return tm_dict

def get_trunc_dict(test_model_list, case=None):

    tm_dict = {}
    for key in test_model_list:
        if any([m in key for m in ['acopf','ptdf','btheta','slopf','plopf']]):
            tm_dict[key] = False
        elif 'lazy' in key:
            tm_dict[key] = False
        else:
            tm_dict[key] = True

    if case is not None:
        replace_unsolved_case(tm_dict, case)

    return tm_dict

def get_error_settings_dict(test_model_list, model='dlopf'):

    tm_list = [tm for tm in test_model_list if model in tm]
    lazy_list = [tm for tm in tm_list if 'lazy' in tm]
    if tm_list != lazy_list:
        tm_list = [tm for tm in tm_list if tm not in lazy_list]

    tm_dict = {}
    for key in test_model_list:
        if key in tm_list:
            tm_dict[key] = True
        else:
            tm_dict[key] = False

    return tm_dict


def replace_unsolved_case(tm_dict, case):

    if 'pglib_opf_' not in case:
        case = 'pglib_opf_' + case
    location = get_solution_file_location(case)
    file_base = os.path.join(location,case)

    for model,include in tm_dict.items():
        if include:
            file_name = file_base + '_' + model + '_1000.json'
            if os.path.exists(file_name):
                continue
            tm_dict[model] = False

            # go through each model and ensure that one is included "True" in the dict
            if 'dlopf' in model:
                for tm in ['dlopf_full','dlopf_lazy_full','dlopf_e4','dlopf_lazy_e4','dlopf_e2','dlopf_lazy_e2']:
                    file_name = file_base + '_' + tm + '_1000.json'
                    if os.path.exists(file_name):
                        tm_dict[tm] = True
                        break
            if 'clopf' in model:
                for tm in ['clopf_full','clopf_lazy_full','clopf_e4','clopf_lazy_e4','clopf_e2','clopf_lazy_e2']:
                    file_name = file_base + '_' + tm + '_1000.json'
                    if os.path.exists(file_name):
                        tm_dict[tm] = True
                        break
            if 'plopf' in model:
                for tm in ['plopf_full','plopf_lazy_full','plopf_e4','plopf_lazy_e4','plopf_e2','plopf_lazy_e2']:
                    file_name = file_base + '_' + tm + '_1000.json'
                    if os.path.exists(file_name):
                        tm_dict[tm] = True
                        break
            if 'ptdf' in model:
                for tm in ['ptdf_full','ptdf_lazy_full','ptdf_e4','ptdf_lazy_e4','ptdf_e2','ptdf_lazy_e2']:
                    file_name = file_base + '_' + tm + '_1000.json'
                    if os.path.exists(file_name):
                        tm_dict[tm] = True
                        break

    return tm_dict

