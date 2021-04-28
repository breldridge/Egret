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

import os, glob, json
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as clrs
import matplotlib.cm as cmap
from matplotlib.colors import ListedColormap
import seaborn as sns
import egret.data.test_utils as tu
import egret.data.lopf_utils as lu
import egret.models.tests.test_approximations as test
from egret.data.model_data import ModelData
from scipy.stats.mstats import gmean,tmean
from egret.parsers.matpower_parser import create_ModelData
#from egret.data.data_utils_deprecated import create_dicts_of_lccm, create_dicts_of_ptdf, create_dicts_of_fdf
import networkx, scipy

# Functions to be summarized by averaging
mean_functions = [tu.num_buses,
                  tu.num_branches,
                  tu.num_constraints,
                  tu.num_variables,
                  tu.num_nonzeros,
                  tu.model_density,
                  tu.solve_time,
                  tu.acpf_slack,
                  tu.balance_slack,
                  tu.vm_viol_sum,
                  tu.thermal_viol_sum,
                  #tu.vm_UB_viol_avg,
                  #tu.vm_LB_viol_avg,
                  #tu.vm_viol_avg,
                  #tu.thermal_viol_avg,
                  tu.vm_UB_viol_max,
                  tu.vm_LB_viol_max,
                  tu.vm_viol_max,
                  tu.thermal_viol_max,
                  #tu.vm_UB_viol_pct,
                  #tu.vm_LB_viol_pct,
                  #tu.vm_viol_pct,
                  #tu.thermal_viol_pct,
                  #tu.thermal_and_vm_viol_pct,
                  tu.pf_error_1_norm,
                  tu.qf_error_1_norm,
                  tu.pf_error_inf_norm,
                  tu.qf_error_inf_norm,
                  tu.total_cost
                  ]

#Functions to be summarized by summation
sum_functions = [tu.optimal,
                 tu.infeasible,
                 tu.maxTimeLimit,
                 tu.maxIterations,
                 tu.solverFailure,
                 tu.internalSolverError,
                 tu.duals,
                 ]

summary_functions = {}
sf = summary_functions
for func in mean_functions:
    key = func.__name__
    sf[key] = {'function' : func, 'dtype' : 'float64'}
    if 'solve_time' in key:
        sf[key]['summarizers'] = ['avg','geomean','max']
    elif 'pct' in key:
        sf[key]['summarizers'] = ['avg','max']
    elif 'avg' in key:
        sf[key]['summarizers'] = ['avg','max']
    elif 'max' in key:
        sf[key]['summarizers'] = ['avg','max']
    elif 'sum' in key:
        sf[key]['summarizers'] = ['avg','sum']
    elif '1_norm' in key:
        sf[key]['summarizers'] = ['sum']
    elif 'inf_norm' in key:
        sf[key]['summarizers'] = ['avg','max']
    else:
        sf[key]['summarizers'] = ['avg']
for func in sum_functions:
    key = func.__name__
    sf[key] = {'function' : func, 'summarizers' : ['sum'], 'dtype' : 'int64'}
    if 'duals' in key:
        sf[key]['dtype'] = 'object'
sf['solve_time']['summarizers'] = ['avg','geomean','max']
sf['acpf_slack']['summarizers'] = ['avg','max']
sf['balance_slack']['summarizers'] = ['avg','sum']

def get_colors(map_name=None, trim=0.9):

    if map_name is None:
        map_name = 'gnuplot'

    trim_top = [
        'ocean',
        'gist_earth',
        'terrain',
        'gnuplot2',
        'CMRmap',
        'cubehelix'
    ]

    colors = cmap.get_cmap(name=map_name)

    if map_name in trim_top:
        trim_colors = ListedColormap(colors(np.linspace(0,trim,256)))
        return trim_colors

    colors.set_bad('grey')

    return colors

def read_data_file(filename="case_data_all.csv"):

    source = lu.get_summary_file_location('data')
    try:
        df_data = pd.read_csv(os.path.join(source, filename), index_col=0)
    except FileNotFoundError:
        df_data = None

    return df_data

def save_data_file(df_data, filename=None):

    if filename is None:
        raise ValueError('Must supply filename in save_data_file.')

    ## save DATA to csv
    print('...out: {}'.format(filename))
    destination = lu.get_summary_file_location('data')
    df_data.to_csv(os.path.join(destination, filename))

def save_figure(filename=None):

    if filename is None:
        raise ValueError('Must supply filename in save_figure.')

    ## save DATA to csv
    print('...out: {}'.format(filename))
    destination = lu.get_summary_file_location('figures')
    plt.savefig(os.path.join(destination, filename))


def read_json_files(test_case):

    _, case = os.path.split(test_case)
    case, _ = os.path.splitext(case)
    case_folder = lu.get_solution_file_location(test_case)
    filename = case + "_*.json"
    file_list = glob.glob(os.path.join(case_folder, filename))

    if len(file_list)==0:
        return None

    data = {}
    for file in file_list:
        md_dict = json.load(open(os.path.join(case_folder, file)))
        md = ModelData(md_dict)
        idx = md.data['system']['filename']
        mult = md.data['system']['mult']
        name = idx.replace(case + '_','')
        name = name.replace('_{0:04.0f}'.format(mult * 1000), '')
        data[idx] = {}
        data[idx]['case'] = case.replace('pglib_opf_','')
        data[idx]['long_case'] = case
        data[idx]['model'] = name
        data[idx]['mult'] = mult
        for col,fdict in summary_functions.items():
            data_generator = fdict['function']
            data[idx][col] = data_generator(md)

        # record lazy setting
        if 'lazy' in name:
            data[idx]['build_mode'] = 'lazy'
        else:
            data[idx]['build_mode'] = 'default'

        # record factor truncation setting, if any
        if any(e in name for e in ['e5','e4','e3','e2']):
            data[idx]['trim'] = name[-2:]
        else:
            data[idx]['trim'] = 'full'

        # record base model
        base_model = name
        setting_list = ['_lazy','_full','_e5','_e4','_e3','_e2']
        remove_list = [s for s in setting_list if s in name]
        for r in remove_list:
            base_model = base_model.replace(r,'')
        data[idx]['base_model'] = base_model

    df_data = pd.DataFrame(data).transpose()
    for col,fdict in summary_functions.items():
        df_data[col] = df_data[col].astype(fdict['dtype'])
    df_data = normalize_solve_time(df_data)
    df_data = normalize_total_cost(df_data)

    return df_data

def update_case_data(case_list, tag=None):

    data = pd.DataFrame(data=None)
    for case in case_list:
        df = read_json_files(case)
        if df is not None:
            data = pd.concat([data, df])
    if data.empty:
        return

    ## save DATA to csv
    filename = "case_data"
    if tag is not None:
        filename += "_" + tag
    filename += ".csv"
    save_data_file(data, filename=filename)


def update_all_case_data(case_sets, tag=None):

    data = pd.DataFrame(data=None)
    for cs in case_sets:
        filename = 'case_data_' + cs + '.csv'
        df = read_data_file(filename=filename)
        if df is not None:
            data = pd.concat([data, df])
    if data.empty:
        return

    ## save DATA to csv
    filename = "case_data_all"
    if tag is not None:
        filename += "_" + tag
    filename += ".csv"
    save_data_file(data, filename=filename)


def nominal_case_data(column, tag=None):

    filename = 'case_data'
    if tag is not None:
        filename += '_' + tag
    df = read_data_file(filename=filename + '.csv')
    if df is None:
        return

    # Remove all rows with mult != 1.000
    df.drop(df[df['mult'] != 1.0].index, inplace=True)

    idx_list = []
    [idx_list.append(x) for x in df['case'].values if x not in idx_list]

    #col_list = []
    #[col_list.append(x) for x in df['model'].values if x not in col_list]
    col_list = ['slopf','dlopf_full','dlopf_lazy_e2','clopf_full','clopf_lazy_e2','plopf_full','plopf_lazy_e2','ptdf_full','ptdf_lazy_e2','btheta']

    data = pd.DataFrame(data=None, index=idx_list, columns=col_list)
    for index, row in df.iterrows():
        idx = row['case']
        col = row['model']
        if col in col_list:
            data.loc[idx, col] = row[column]

    data.sort_index(axis=1, inplace=True)

    ## save DATA to csv
    filename += '_' + column + '.csv'
    save_data_file(data, filename=filename)


def nominal_all_data(case_keys, column='solve_time', function='gm'):

    data = pd.DataFrame(data=None, index=case_keys)

    for tag in case_keys:
        filename = 'case_data' + '_' + tag + '_' + column
        df = read_data_file(filename=filename + '.csv')
        if df is None:
            continue

        for col in df.columns:
            arr = df[col].dropna(0).values
            if function=='gm':
                f = gmean(arr)
            elif function=='tm':
                f = tmean(arr)
            elif function=='max':
                f = max(arr)
            else:
                message = '{} not accepted. Use gm, tm, or max.'.format(function)
                raise ValueError(message)
            data.loc[tag, col] = f

    ## save DATA to csv
    filename = 'set_data_' + column + '_' + function + '.csv'
    save_data_file(data, filename=filename)


def filter_dataframe(df_data, data_filters=None):

    if df_data is None:
        return None
    if data_filters is not None:
        for col,keepers in data_filters.items():
            if col in df_data.columns:
                drop_rows = []
                for idx,row in df_data.iterrows():
                    if not any(k in row[col] for k in keepers):
                        drop_rows.append(idx)
                df_data = df_data.drop(drop_rows)

    return df_data


def format_model_desc(obj, column_name=None):

    def make_replacements(name):
        name = name.capitalize()
        name = name.replace('_full','')
        name = name.replace('_',', ')
        name = name.replace('lopf', '-LOPF')
        name = name.replace('Ptdf', 'PTDF')
        name = name.replace('Btheta', 'B-theta')
        name = name.replace('Acopf', 'AC OPF')
        return name

    if isinstance(obj, pd.DataFrame):
        if column_name is None:
            cols = [make_replacements(c) for c in obj.columns]
            obj.columns = cols
            return obj
        else:
            for i in obj.index:
                obj.loc[i, column_name] = make_replacements(obj.loc[i,column_name])
            return obj

    elif isinstance(obj, list):
        new_list = [make_replacements(name) for name in obj]
        return new_list



def generate_boxplot(data_filters=None, data_name='solve_time', data_unit=None, order=None, category='model', hue=None,
                         scale='linear', filename=None, sns_plot=sns.boxplot, palette=None, show_plot=True):

    if filename is None:
        filename = "all_summary_data.csv"
    df_data = read_data_file(filename=filename)
    df_data = filter_dataframe(df_data, data_filters)
    if df_data is None:
        return
    df_data = format_model_desc(df_data, category)
    order = format_model_desc(order)

    if hue=='trim':
        hue_order = ['full', 'e4', 'e2']
    elif hue=='build_mode':
        hue_order = ['default', 'lazy']
    else:
        hue_order = None

    settings = {}
    settings['hue'] = hue
    settings['order'] = order
    settings['hue_order'] = hue_order
    settings['palette'] = palette
    #settings['inner'] = 'quartile'
    #settings['scale'] = 'width'
    #settings['color'] = '0.8'
    #settings['orient'] = 'h'
    #settings['bw'] = 0.2
    #ax = sns.violinplot(y=category, x=data_name, data=df_data, **settings)
    #ax = sns.stripplot(y=category, x=data_name, data=df_data, order=order, dodge=True, size=2.5)
    #ax = sns.boxplot(y=category, x=data_name, data=df_data, **settings)
    sns.set_theme(style="ticks")
    ax = sns_plot(y=category, x=data_name, data=df_data, **settings)
    ax.set_ylabel(category)
    if data_unit is None:
        ax.set_xlabel(data_name)
    else:
        ax.set_xlabel(data_name + '('+data_unit+')')
    #ax.set_yticklabels(ax.get_yticklabels(), rotation=90)
    #ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    plt.tight_layout()
    plt.xscale(scale)
    sns.despine()

    ## save FIGURE as png
    tag = data_filters['file_tag']
    filename = "boxplot_" + data_name + "_" + tag + ".png"
    save_figure(filename=filename)

    if show_plot:
        plt.show()
    else:
        plt.close('all')


def generate_boxplots(show_plot=False):

    # lists used for filtering categorical labels
    model_list = ['slopf','dlopf','clopf','plopf','ptdf','btheta']
    dense_list = ['dlopf','clopf','plopf','ptdf']
    case_sets = ['all','ieee','k','rte','sdet','tamu','pegase','misc']

    for cs in case_sets:

        # default mode results
        filename = 'case_data_' + cs + '.csv'
        filters = {}
        #filters['base_model'] = ['acopf','slopf','dlopf','clopf','plopf','ptdf','btheta']
        #filters['build_mode'] = ['default']
        #filters['trim'] = ['full']
        #filters['file_tag'] = cs + '_default_options'
        filters['file_tag'] = cs + '_allbuilds'
        model_order = test.get_test_model_list()
        generate_boxplot(data_name='solve_time', data_unit='s', data_filters=filters, category='model', order=model_order,
                         filename=filename, scale='log', sns_plot=sns.stripplot, palette='gnuplot', show_plot=show_plot)

        # default/lazy mode results
        filename = 'case_data_' + cs + '.csv'
        filters = {}
        filters['base_model'] = ['slopf','dlopf','clopf','plopf']
        filters['trim'] = ['full']
        filters['file_tag'] = cs + '_lazy_options'
        generate_boxplot(data_name='solve_time', data_unit='s', data_filters=filters, category='base_model', order=filters['base_model'],
                         filename=filename, scale='linear', hue='build_mode', palette='gnuplot2', show_plot=show_plot)

        # factor tolerance results
        filename = 'case_data_' + cs + '.csv'
        filters = {}
        filters['base_model'] = ['slopf','dlopf','clopf','plopf']
        filters['build_mode'] = ['default']
        filters['file_tag'] = cs + '_trim_options'
        generate_boxplot(data_name='solve_time', data_unit='s', data_filters=filters, category='base_model', order=filters['base_model'],
                         filename=filename, scale='linear', hue='trim', palette='gnuplot2', show_plot=show_plot)

        # factor tolerance results
        filename = 'case_data_' + cs + '.csv'
        filters = {}
        filters['model'] = ['acopf','slopf','dlopf_full','dlopf_lazy_e2','clopf_full','clopf_lazy_e2','plopf_full','plopf_lazy_e2','ptdf_full','ptdf_lazy_e2','btheta']
        filters['file_tag'] = cs + '_compare_simplified'
        generate_boxplot(data_name='solve_time', data_unit='s', data_filters=filters, category='model', order=filters['model'],
                         filename=filename, scale='linear', sns_plot=sns.boxplot, palette='gnuplot2', show_plot=show_plot)




def generate_network_data(test_case, test_model_list, data_generator=None):

    case_location = lu.get_solution_file_location(test_case)
    src_folder, case_name = os.path.split(test_case)
    case_name, ext = os.path.splitext(case_name)

    if data_generator is None:
        data_generator = tu.thermal_viol
    func_name = data_generator.__name__

    df_data = pd.DataFrame(data=None)
    for test_model in test_model_list:
        try:
            df_raw = read_nominal_data(case_location, test_model, data_generator=data_generator)
        except:
            df_raw = pd.DataFrame(data=None, index=[test_model])
        df_data = pd.concat([df_data, df_raw], sort=True)
    df_data = df_data.transpose()

    ## save DATA to csv
    filename = func_name + '_' + case_name + ".csv"
    save_data_file(df_data, filename=filename)


def generate_barplot(test_case, test_model_dict=None, data_name=None, units=None, file_tag=None, show_plot=False):

    if data_name is None:
        data_name = 'thermal_viol'
    if units is None:
        units = data_name
    else:
        units = data_name + " (" + units + ")"

    def help_get_data(case):
        src_folder, case_name = os.path.split(case)
        case_name, ext = os.path.splitext(case_name)
        filename = data_name + "_" + case_name + ".csv"
        df_data = get_data(filename, test_model_dict=test_model_dict)
        if df_data.empty:
            return
        return df_data, case_name

    if isinstance(test_case, list):
        df_data = pd.DataFrame(data=None)
        for case in test_case:
            _df, _ = help_get_data(case)
            df_data = df_data.append(_df, ignore_index=True)
        case_name = 'summary'
    else:
        df_data, case_name = help_get_data(test_case)

    # median, mean, and maximum error statistics
    models = ['slopf','dlopf','clopf','plopf','ptdf','btheta']
    trims = ['full', 'e4', 'e2']
    labels = ['model','median','mean','max','trim']
    summary = pd.DataFrame(data=None, columns=labels)
    data = df_data.fillna(0).abs()
    for c in df_data.columns:
        row = {}
        model = [m for m in models if m in c][0]
        if model in ['dlopf','clopf','plopf','ptdf']:
            trim = [t for t in trims if t in c][0]
        else:
            trim = 'full'
        row['model'] = model
        row['median'] = data[c].median()
        row['mean'] = data[c].mean()
        row['max'] = data[c].max()
        row['trim'] = trim
        summary = summary.append(row, ignore_index=True)
    summary = format_model_desc(summary, 'model')

    ## Create plots
    stats = ['median', 'mean', 'max']
    for stat in stats:
        sns.set(style="whitegrid")
        ax = sns.barplot(x='model', y=stat, hue='trim', data=summary)
        ax.set_yscale("log")
        tick_loc = mpl.ticker.LogLocator(base=10.0, subs='all')
        tick_fmt = mpl.ticker.LogFormatterSciNotation(base=10.0, labelOnlyBase=False, minor_thresholds=(2,0))
        ax.yaxis.set_minor_locator(tick_loc)
        ax.yaxis.set_minor_formatter(tick_fmt)
        ax.yaxis.set_major_formatter(tick_fmt)
        plt.grid(True, which="minor", axis="y", ls=":", c=[0.8,0.8,0.8])
        ax.set_xlabel("Model")
        ax.set_ylabel(units)
        plt.tight_layout()
        plt.legend(loc="upper left")

        ## save FIGURE as png
        filename = case_name + "_" + data_name + "_barplot_" + stat
        if file_tag is not None:
            filename += "_" + file_tag
        save_figure(filename+".png")

        if show_plot:
            plt.show()
        else:
            plt.close('all')


def generate_heatmap(test_case, test_model_dict=None, data_name=None, index_name=None, units=None, N=None,
                               file_tag=None, colormap=None, show_plot=False):

    src_folder, case_name = os.path.split(test_case)
    case_name, ext = os.path.splitext(case_name)

    if data_name is None or data_name is 'thermal_viol':
        data_name = 'thermal_viol'
        colormap = ListedColormap(colormap(np.linspace(0.5, 1, 256)))
        colormap.set_bad('grey')
    if index_name is None:
        index_name = 'Branch'
    if units is None:
        units = 'MW'

    filename = data_name + "_" + case_name + ".csv"
    df_data = get_data(filename, test_model_dict=test_model_dict)
    df_data = format_model_desc(df_data)
    if df_data.empty:
        return

    # sort by average absolute errors and display at most N rows
    data = df_data.fillna(0)
    if N is not None:
        df_data['abs_max'] = data.abs().max(axis=1)
        df_data = df_data.sort_values('abs_max', ascending=False)
        df_data = df_data.drop('abs_max', axis=1)
        df_data = df_data.head(50)
        vmin = min(data.values.min(), -0.001)
        vmax = max(data.values.max(), 0.001)
    else:
        vmin = data.values.min()
        vmax = data.values.max()

    kwargs={}
    cbar_dict = {}
    cbar_dict['label'] = data_name + ' (' + units + ')'
    if data_name not in ['thermal_viol','lmp']:
        kwargs['vmin'] = min(vmin,-vmax)
        kwargs['vmax'] = max(vmax,-vmin)
    elif data_name == 'lmp':
        kwargs['vmin'] = max(vmin,-250)
        kwargs['vmax'] = min(vmax, 250)
    kwargs['linewidth'] = 0
    kwargs['cmap'] = colormap
    kwargs['cbar_kws'] = cbar_dict

    # Create heatmap in Seaborn
    ## Create plot
    #plt.figure(figsize=(5.5, 8.5))
    ax = sns.heatmap(df_data, **kwargs)

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    #ax.set_title(case_name + " " + viol_name)
    ax.set_xlabel("Model")
    ax.set_ylabel(index_name)
    #ax.set_yticks([])

    plt.tight_layout()

    ## save FIGURE as png
    filename = case_name + "_" + data_name
    if file_tag is not None:
        filename += "_" + file_tag
    filename += ".png"
    print('...out: {}'.format(filename))
    destination = lu.get_summary_file_location('figures')
    plt.savefig(os.path.join(destination, filename))

    if show_plot:
        plt.show()
    else:
        plt.close('all')


def generate_plot_data(case_list):
    # Generate data
    data_list = [tu.pf_error, tu.qf_error, tu.thermal_viol, tu.vm_viol, tu.lmp]
    tml = test.get_test_model_list()
    for c in case_list:
        for d in data_list:
            generate_network_data(c, tml, data_generator=d)


def generate_barplots(case_list=None, list_name=None, show_plot=False):

    if case_list is None:
        case_list = test.get_case_names()
    tml = test.get_test_model_list()

    model_dict = lu.get_barplot_dict(tml)
    generate_barplot(case_list, test_model_dict=model_dict, data_name='pf_error', units='MW', file_tag=list_name, show_plot=show_plot)

    for c in case_list:
        model_dict = lu.get_barplot_dict(tml, case=c)
        generate_barplot(c, test_model_dict=model_dict, data_name='pf_error', units='MW', show_plot=False)


def generate_heatmaps(case_list=None, colors=None, show_plot=False):

    if case_list is None:
        case_list = test.get_case_names()
    if colors is None:
        colors = get_colors('coolwarm')
        lmp_colors = get_colors('bone')
    tml = test.get_test_model_list()

    for c in case_list:
        model_dict = lu.get_vanilla_dict(tml, case=c)
        generate_heatmap(c, test_model_dict=model_dict, data_name='pf_error', N=50, file_tag='vanilla',
                         index_name='Branch', units='MW', colormap=colors, show_plot=show_plot)

        model_dict['acopf'] = True
        generate_heatmap(c, test_model_dict=model_dict, data_name='lmp', N=None, file_tag='vanilla',
                         index_name='Bus', units='$/MWh', colormap=lmp_colors, show_plot=show_plot)

        model_dict = lu.get_lazy_dict(tml)
        generate_heatmap(c, test_model_dict=model_dict, data_name='pf_error', N=50, file_tag='lazy',
                         index_name='Branch', units='MW', colormap=colors, show_plot=show_plot)

        model_dict = lu.get_trunc_dict(tml)
        generate_heatmap(c, test_model_dict=model_dict, data_name='pf_error', N=50, file_tag='trunc',
                         index_name='Branch', units='MW', colormap=colors, show_plot=show_plot)


def generate_sensitivity(test_case, model_dict, y_data='acpf_slack', y_units='MW',
                           colors=None, show_plot=True):

    _, case_name = os.path.split(test_case)
    case_name, ext = os.path.splitext(case_name)

    df = read_data_file('case_data_all.csv')

    # filter by test case and model
    model_list = [k for k,v in model_dict.items() if v]
    df = df[df.long_case==case_name]
    df = df[df.model.isin(model_list)]

    if len(df.index) <= len(model_list):
        print('...skipping sensitivity graph due to lack of data.')
        return

    # create empty dataframe
    idx_list = list(set(df.mult.values))
    idx_list.sort()
    df_data = pd.DataFrame(data=None, index=idx_list, columns=model_list)

    # fill dataframe with data
    for idx,row in df.iterrows():
        model = row.model
        x = row.mult
        y = row[y_data]
        df_data.loc[x,model] = y

    ## save DATA to csv
    filename = "sensitivity_" + case_name + "_" + y_data + ".csv"
    save_data_file(df_data,filename=filename)

    ## Create plot
    fig, ax = plt.subplots()

    #---- set properties
    if colors is None:
        colors = get_colors('cubehelix', trim=0.8)
    marker_style = pareto_marker_style(model_list, colors=colors)

    # plot
    for m in model_list:
        ms = marker_style[m]
        ms['linestyle'] = 'solid'
        ms['markeredgewidth'] = 1.5
        if 'slopf' in m or 'btheta' in m:
            ms['marker'] = 's'
        elif 'dlopf' in m:
            ms['marker'] = 'o'
        elif 'clopf' in m or 'ptdf' in m:
            ms['marker'] = 'x'
        elif 'plopf' in m:
            ms['marker'] = '+'
        else:
            ms['marker'] = None
        ms['fillstyle'] = 'none'
        x = idx_list
        y = df_data[m]
        ax.plot(x, y, **ms)

    # formatting
    ylabel = y_data
    if y_units is not None:
        ylabel += " (" + y_units +")"
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Demand Multiplier')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, 0.8 * box.width, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ## save FIGURE as png
    filename = "sensitivity_" + case_name + "_" + y_data + ".png"
    save_figure(filename)

    # display
    if show_plot is True:
        plt.show()
    else:
        plt.close('all')


def generate_sensitivities(case_list, show_plot=False):

    tml = test.get_test_model_list()
    model_dict = lu.get_sensitivity_dict(tml)

    for case in case_list:
        generate_sensitivity(case, model_dict, y_data='acpf_slack', y_units='MW', show_plot=show_plot)
        # Note: 'total_cost_normalized' is only available if the acopf solve was successful
        generate_sensitivity(case, model_dict, y_data='total_cost_normalized', y_units=None, show_plot=show_plot)
        generate_sensitivity(case, model_dict, y_data='pf_error_1_norm', y_units='MW', show_plot=show_plot)
        generate_sensitivity(case, model_dict, y_data='pf_error_inf_norm', y_units='MW', show_plot=show_plot)

def summarize_sensitivity(data_name, flag=None, show_plot=False):

    case_list = test.get_case_names(flag=flag)
    if 'pglib_opf_case300_ieee' in case_list:
        case_list.remove('pglib_opf_case300_ieee')

    df = pd.DataFrame(data=None)
    for case in case_list:
        filename = 'sensitivity_' + case + '_' + data_name + '.csv'
        _df = read_data_file(filename)
        if 'acopf' in _df.columns:
            _df.drop('acopf', axis=1, inplace=True)
        _df = _df.melt(var_name='model', value_name=data_name)
        _df['case'] = case[10:]
        df = pd.concat([df, _df])

    if flag is not None:
        filename = 'sensitivity_pglib_opf_' + flag + '_' + data_name + '.csv'
    else:
        filename = 'sensitivity_pglib_opf_' + data_name + '.csv'
    save_data_file(df, filename=filename)

    sns.set(style="whitegrid")
    ax = sns.barplot(x='model', y=data_name, hue='case', data=df)
    ax.set_ylabel(data_name)
    ax.set_xlabel('Model')

    if data_name == 'total_cost_normalized':
        ax.set(ylim=(0.99, 1.006))

    filename = "barplot_" + data_name
    if flag is not None:
        filename += "_" + flag
    save_figure(filename=filename + ".png")

    if show_plot:
        plt.show()
    else:
        plt.close('all')


def summarize_sensitivities(flag=None, show_plot=False):

    stats = ['total_cost_normalized', 'acpf_slack', 'pf_error_1_norm', 'pf_error_inf_norm']
    for s in stats:
        summarize_sensitivity(s, flag=flag, show_plot=show_plot)

def update_data_file(test_case):

    try:
        df1 = read_data_file(filename='case_data_all.csv')
    except:
        df1 = pd.DataFrame(data=None)

    df2 = read_json_files(test_case)

    # merge new data
    new_index = list(set(list(df1.index.values) + list(df2.index.values)))
    new_columns = list(set(list(df1.columns.values) + list(df2.columns.values)))
    update = df1.reindex(index=new_index,columns=new_columns)
    for idx,row in df2.iterrows():
        update.loc[idx] = row
    update.sort_index(inplace=True)

    ## save DATA to csv
    destination = lu.get_summary_file_location('data')
    filename = "all_summary_data.csv"
    print('...out: {}'.format(filename))
    update.to_csv(os.path.join(destination, filename))


def create_table1_solverstatus(df, tag=None):

    model_list = list(set(list(df.model.values)))
    desc_list = ['base_model','trim','build_mode']
    count_list = ['optimal','infeasible','duals','internalSolverError','maxIterations','maxTimeLimit','solverFailure']
    df1 = pd.DataFrame(data=None, index=model_list, columns=desc_list+count_list)

    # Table 1
    for idx,row in df1.iterrows():
        _df = df[df.model==idx]
        dummy_row = _df.iloc[0]
        for col in desc_list:
            row[col] = dummy_row[col]
        for col in count_list:
            _col = _df[col]
            _col[_col=='True'] = 1
            _col[_col=='False'] = 0
            row[col] = sum(_col.astype(int))
    df1 = df1.sort_values(by=['base_model','build_mode','trim'], ascending=[1,1,0])

    ## save DATA to csv
    filename = "table1_solverstatus"
    if tag is not None:
        filename += '_' + tag
    filename += ".csv"
    save_data_file(df1, filename=filename)

def create_table2_timeandcost(df, tag=None):

    model_list = list(set(list(df.model.values)))
    desc_list = ['base_model','trim','build_mode']
    gmean_list = ['solve_time_normalized','total_cost_normalized','acpf_slack','num_variables','num_constraints','num_nonzeros']
    df2 = pd.DataFrame(data=None, index=model_list, columns=desc_list+gmean_list)

    # Table 2
    for idx, row in df2.iterrows():
        _df = df[df.model == idx]
        dummy_row = _df.iloc[0]
        for col in desc_list:
            row[col] = dummy_row[col]
        for col in gmean_list:
            df_nz = _df[_df[col]!=0]
            array = abs(df_nz[col].dropna())
            row[col] = gmean(array)
    df2 = df2.sort_values(by=['base_model','build_mode','trim'], ascending=[1,1,0])

    ## save DATA to csv
    filename = "table2_timeandcost"
    if tag is not None:
        filename += '_' + tag
    filename += ".csv"
    save_data_file(df2, filename=filename)

def create_table3_violations(df, tag=None):

    model_list = list(set(list(df.model.values)))
    desc_list = ['base_model','trim','build_mode']
    tmean_list = ['vm_viol_sum','vm_viol_max','thermal_viol_sum','thermal_viol_max']
    max_list = tmean_list
    df3_list = [c + '_avg' for c in tmean_list] + [c + '_max' for c in max_list]
    df3 = pd.DataFrame(data=None, index=model_list, columns=desc_list+df3_list)

    # Table 3
    for idx, row in df3.iterrows():
        _df = df[df.model == idx]
        dummy_row = _df.iloc[0]
        for col in desc_list:
            row[col] = dummy_row[col]
        for col in tmean_list:
            row[col + '_avg'] = tmean(_df[col].dropna())
        for col in max_list:
            row[col + '_max'] = max(_df[col])
    df3 = df3.sort_values(by=['base_model','build_mode','trim'], ascending=[1,1,0])

    ## save DATA to csv
    filename = "table3_violations"
    if tag is not None:
        filename += '_' + tag
    filename += ".csv"
    save_data_file(df3, filename=filename)


def create_table(df_in, col_name, max_buses=None, min_buses=None, tag=None):

    model_list = list(set(list(df_in.model.values)))
    desc_list = ['base_model','trim','build_mode']
    case_list = lu.case_names
    df_out = pd.DataFrame(data=None, index=model_list, columns=desc_list+case_list)

    # Fill in table
    for idx,row in df_out.iterrows():
        _df = df_in[df_in.model==idx]
        dummy_row = _df.iloc[0]
        for col in desc_list:
            row[col] = dummy_row[col]
        for case in case_list:
            _case = _df[_df.long_case==case]
            if len(_case.index) == 1:
                row[case] = _case.iloc[0][col_name]
            elif len(_case.index) > 1:
                _case = _case[_case.mult==1]
                row[case] = _case.iloc[0][col_name]
            else:
                pass
    df_out = df_out.sort_values(by=['base_model','build_mode','trim'], ascending=[1,1,0])

    # add short case name and num_bus to top two rows
    df_top = pd.DataFrame(data=None, index=['case', 'num_buses'], columns=case_list)
    for case in case_list:
        _df = df_in[df_in.long_case==case]
        if not _df.empty:
            dummy_row = _df.iloc[0]
            df_top.loc['case',case] = dummy_row['case']
            df_top.loc['num_buses',case] = dummy_row['num_buses']
    df_out = pd.concat([df_top, df_out], sort=False)

    # Find the columns where each value is null and drop from the dataframe
    empty_cols = [col for col in df_out.columns if df_out[col].isnull().all()]
    df_out.drop(empty_cols, axis=1, inplace=True)

    filename = 'table_' + col_name
    if max_buses is not None:
        filename += '_under_{}'.format(max_buses)
        drop_col = [c for c in df_out.columns if df_out.loc['num_buses',c] <= max_buses]
        df_out = df_out.drop(drop_col, axis='columns')
    if min_buses is not None:
        filename += '_over_{}'.format(min_buses)
        drop_col = [c for c in df_out.columns if df_out.loc['num_buses',c] >= min_buses]
        df_out = df_out.drop(drop_col, axis='columns')
    if tag is not None:
        filename += '_' + tag

    ## save DATA to csv
    destination = lu.get_summary_file_location('data')
    filename += ".csv"
    print('...out: {}'.format(filename))
    df_out.to_csv(os.path.join(destination, filename))


def update_data_tables(tag=None):

    filename = 'case_data'
    if tag is not None:
        filename += '_' + tag
    df = read_data_file(filename=filename + '.csv')
    if df is None:
        return

    # Tables
    create_table1_solverstatus(df, tag=tag)
    create_table2_timeandcost(df, tag=tag)
    create_table3_violations(df, tag=tag)

    # takes geomean of column in df
    create_table(df,'acpf_slack', tag=tag)
    create_table(df,'balance_slack', tag=tag)
    create_table(df,'solve_time', tag=tag)
    create_table(df,'total_cost_normalized', tag=tag)

    # takes nominal value of column in df
    nominal_case_data('total_cost_normalized', tag=tag)
    nominal_case_data('solve_time', tag=tag)
    nominal_case_data('speedup', tag=tag)

def normalize_solve_time(df_data, model_benchmark='acopf'):

    df_benchmark = df_data[df_data.model==model_benchmark]
    df_benchmark = df_benchmark.select_dtypes(include='number')
    arr = list(df_benchmark.solve_time.values)
    arr = [a for a in arr if a != 0]
    gm_benchmark = gmean(arr)

    df_data['solve_time_normalized'] = df_data['solve_time'] / gm_benchmark
    df_data['speedup'] = gm_benchmark / df_data['solve_time']

    return df_data

def normalize_total_cost(df_data, model_benchmark='acopf'):

    df_data = df_data.sort_values(by='mult')
    is_benchmark = df_data['model']==model_benchmark
    df_benchmark = df_data[is_benchmark]

    model_list = list(set(df_data.model.values))
    model_list.sort()

    for m in model_list:
        df_m = df_data[df_data.model==m]
        for idx,row in df_m.iterrows():
            mult = row.mult
            tc1 = row.total_cost
            df2 = df_benchmark[df_benchmark.mult==mult]
            tc2 = df2.total_cost.values.item()
            try:
                val = tc1 / tc2
            except:
                val = None
            df_data.loc[idx, 'total_cost_normalized'] = val

    return df_data


def read_solution_data(case_folder, test_model, data_generator=tu.solve_time):
    parent, case = os.path.split(case_folder)
    filename = case + "_" + test_model + "_*.json"
    file_list = glob.glob(os.path.join(case_folder, filename))

    data = {}
    data_type = data_generator.__name__
    for file in file_list:
        md_dict = json.load(open(os.path.join(case_folder, file)))
        md = ModelData(md_dict)
        idx = md.data['system']['filename']
        data[idx] = {}
        data[idx]['case'] = case
        data[idx]['model'] = test_model
        data[idx]['mult'] = md.data['system']['mult']
        data[idx][data_type] = data_generator(md)

    df_data = pd.DataFrame(data).transpose()

    return df_data

def read_nominal_data(case_folder, test_model, data_generator=tu.thermal_viol):
    parent, case = os.path.split(case_folder)
    ## assumed that detailed data is only needed for the nominal demand case
    filename = case + "_" + test_model + "_1000.json"

    try:
        md_dict = json.load(open(os.path.join(case_folder, filename)))
    except:
        return pd.DataFrame(data=None, index=[test_model])

    md = ModelData(md_dict)
    data = data_generator(md)
    cols = list(data.keys())
    new_cols = [int(c) for c in cols]
    df_data = pd.DataFrame(data, index=[test_model])
    df_data.columns = new_cols

    return df_data


def generate_speedup_data(case_list=None, mean_data='solve_time_geomean', benchmark='dlopf_lazy'):

    ## get data
    if case_list is None:
        case_list = lu.case_names[:]

    data_dict = {}
    cases = []
    for case in case_list:
        try:
            input = "summary_data_" + case + ".csv"
            df_data = get_data(input)
            models = list(df_data.index.values)
            for m in models:
                val = df_data.at[benchmark, mean_data] / df_data.at[m, mean_data]
                if m in data_dict:
                    data_dict[m].append(val)
                else:
                    data_dict[m] = [val]
            cases.append(case)
        except:
            pass

    df_data = pd.DataFrame(data_dict,index=cases)
    df_data.loc['AVERAGE'] = df_data.mean()

    ## save DATA to csv
    destination = lu.get_summary_file_location('data')
    filename = "speedup_data_" + mean_data + "_" + benchmark + ".csv"
    df_data.to_csv(os.path.join(destination, filename))


def generate_speedup_heatmap(test_model_dict=None, mean_data='solve_time_geomean', benchmark='dlopf_lazy',colormap=None,
                             cscale='linear', include_benchmark=False, show_plot=False):

    filename = "speedup_data_" + mean_data + "_" + benchmark + ".csv"
    df_data = get_data(filename, test_model_dict=test_model_dict)
    if not include_benchmark and benchmark in df_data.columns.to_list():
        df_data = df_data.drop(columns=benchmark)

    cols = df_data.columns.to_list()
    col_lazy=[]
    col_alert=[]
    for c in cols:
        if 'lazy' in c:
            col_lazy.append(c)
        else:
            col_alert.append(c)
    cols = col_alert + col_lazy
    df_data = df_data[cols]

    model_names = [c for c in df_data.columns]
#    index_names = [i for i in df_data.index]
    index_names = [i.replace('pglib_opf_','') for i in df_data.index]
    data = df_data.values
    model_num = len(model_names)

    #   EDIT TICKS HERE IF NEEDED   #
    if cscale == 'log':
        cbar_dict = {'ticks' : [1e0,1e1,1e2]}
        cbar_norm = clrs.LogNorm(vmin=data.min(), vmax=data.max())
    else:
        cbar_dict = None
        cbar_norm = None

    ax = sns.heatmap(data,
                     linewidth=0.,
                     xticklabels=model_names,
                     yticklabels=index_names,
                     cmap=colormap,
                     norm=cbar_norm,
                     cbar_kws=cbar_dict,
                     )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    ax.set_title(mean_data + " speedup vs. " + benchmark)
    ax.set_xlabel("Model")
    ax.set_ylabel("Test Case")

    plt.tight_layout()

    ## save FIGURE as png
    filename = "speedupplot_v_" + benchmark + "_" + mean_data + ".png"
    destination = lu.get_summary_file_location('figures')
    plt.savefig(os.path.join(destination, filename))

    if show_plot:
        plt.show()
    else:
        plt.close('all')


def generate_sensitivity_data(test_case, test_model_list, data_generator=tu.acpf_slack,
                              data_is_pct=False, data_is_vector=False, vector_norm=2):

    case_location = lu.get_solution_file_location(test_case)
    src_folder, case_name = os.path.split(test_case)
    case_name, ext = os.path.splitext(case_name)

    # acopf comparison
    df_acopf = read_sensitivity_data(case_location, 'acopf', data_generator=data_generator)


    ## calculates specified L-norm of difference with acopf (e.g., generator dispatch, branch flows, voltage profile...)
    if data_is_vector:
        print('data is vector of length {}'.format(len(df_acopf.values)))

    ## calcuates relative difference from acopf (e.g., objective value, solution time...)
    elif data_is_pct:
        acopf_avg = sum(df_acopf[idx].values for idx in df_acopf) / len(df_acopf)
        print('data is pct with acopf values averaging {}'.format(acopf_avg))
    else:
        acopf_avg = sum(df_acopf[idx].values for idx in df_acopf) / len(df_acopf)
        print('data is nominal with acopf values averaging {}'.format(acopf_avg))

    # empty dataframe to add data into
    df_data = pd.DataFrame(data=None)

    # iterate over test_models
    if 'acopf'  not in test_model_list:
        test_model_list.append('acopf')

    for test_model in test_model_list:
        df_approx = read_sensitivity_data(case_location, test_model, data_generator=data_generator)

        # calculate norm from df_diff columns
        data = {}
        avg_ac_data = sum(df_acopf[idx].values for idx in df_acopf) / len(df_acopf)
        for col in df_approx:
            if data_is_vector is True:
                data[col] = np.linalg.norm(df_approx[col].values - df_acopf[col].values, vector_norm)
            elif data_is_pct is True:
                data[col] = ((df_approx[col].values - df_acopf[col].values) / df_acopf[col].values) * 100
            else:
                data[col] = df_approx[col].values

        # record test_model column in DataFrame
        df_col = pd.DataFrame(data, index=[test_model])
        df_data = pd.concat([df_data, df_col], sort=True)


    ## save DATA as csv
    y_axis_data = data_generator.__name__
    df_data = df_data.T
    destination = lu.get_summary_file_location('data')
    filename = "sensitivity_data_" + case_name + "_" + y_axis_data + ".csv"
    df_data.to_csv(os.path.join(destination, filename))

def generate_pareto_all_data(test_model_list, data_column='solve_time_normalized', mean_type='geomean'):

    ## Pull all current summar_data_*.csv files
    data_location = lu.get_summary_file_location('data')
    filename = "summary_data_*.csv"
    file_list = glob.glob(os.path.join(data_location, filename))

    df_x = pd.DataFrame(data=None)
    for file in file_list:
        src_folder, case_name = os.path.split(file)
        case_name, ext = os.path.splitext(case_name)
        case_name = case_name.replace('summary_data_pglib_opf_','')

        df_raw = get_data(file)
        df_x[case_name] = df_raw[data_column]

    ## Add last colunm
    #   - try to use 'geomean' for normalized data (e.g. solve time) and 'avg' for arithmetic data
    index_list = list(df_x.index)
    #masked_data = ma.masked_array(df_x.values, mask=df_x.isnull())

    # remove test case if any model was AC-infeasible
    nullcol = df_x.columns[df_x.isnull().any()].to_list()
    clean_data = df_x.drop(nullcol, axis=1)
    if mean_type is 'geomean':
        mean_data = gmean(clean_data, axis=1)
    if mean_type is 'avg':
        mean_data = tmean(clean_data, axis=1)
    df_x[data_column] = mean_data

    ## save DATA as csv
    destination = lu.get_summary_file_location('data')
    filename = "pareto_all_data_" + data_column + ".csv"
    df_x.to_csv(os.path.join(destination, filename))

def pareto_marker_style(model_list, colors=cmap.viridis):
    # Creates a dict of dicts of marker styles for each model
    n = len(model_list)
    color_list = [colors(i) for i in np.linspace(0, 1, n)]
    color_dict = dict(zip(model_list,color_list))

    marker_style = {}
    ms = marker_style
    for m in model_list:
        ms[m] = dict(linestyle='', markersize=8)
        fmt = ms[m]
        fmt['label'] = m
        fmt['color'] = color_dict[m]
        if 'acopf' in m:
            fmt['marker'] = '*'

        elif 'full' in m:
            fmt['marker'] = 's'
            fmt['markeredgewidth'] = 2
            fmt['fillstyle'] = 'none'

        elif '_e5' in m:
            fmt['marker'] = '.'
            fmt['markeredgewidth'] = 2
            fmt['fillstyle'] = 'none'

        elif '_e4' in m:
            fmt['marker'] = 'o'
            fmt['markeredgewidth'] = 2
            fmt['fillstyle'] = 'none'

        elif '_e3' in m:
            fmt['marker'] = 'x'
            fmt['markeredgewidth'] = 2
            fmt['fillstyle'] = 'none'

        elif '_e2' in m:
            fmt['marker'] = '+'
            fmt['markeredgewidth'] = 2
            fmt['fillstyle'] = 'none'

        elif 'btheta' in m:
            if 'qcp' in m:
                fmt['marker'] = '$Q$'
            else:
                fmt['marker'] = '$B$'

        elif 'slopf' in m:
            fmt['marker'] = '$S$'

    return marker_style

def generate_pareto_all_plot(test_model_dict,
                             y_data = 'thermal_viol_sum_avg', x_data = 'solve_time_normalized',
                             y_units = 'MW', x_units = None, colors = cmap.viridis,
                             annotate_plot=False, show_plot = False):
    ## get data
    data_location = lu.get_summary_file_location('data')
    file_x = os.path.join(data_location, "pareto_all_data_" + x_data + ".csv")
    file_y = os.path.join(data_location, "pareto_all_data_" + y_data + ".csv")

    df_raw = get_data(file_x, test_model_dict=test_model_dict)
    df_x_data = pd.DataFrame(df_raw[x_data])
    df_raw = get_data(file_y, test_model_dict=test_model_dict)
    df_y_data = pd.DataFrame(df_raw[y_data])

    models = list(df_raw.index.values)

    ## Create plot
    fig, ax = plt.subplots()

    #---- set property cycles
    marker_style = pareto_marker_style(models, colors=colors)

    for m in models:
        ms = marker_style[m]
        x = df_x_data.loc[m]
        y = df_y_data.loc[m]
        ax.plot(x, y, **ms)

        if annotate_plot:
            ax.annotate(m, (x,y))

    ax.set_title(y_data + " vs. " + x_data + "\n(all test cases)")
    y_label = y_data
    x_label = x_data
    if y_units is not None:
        y_label += " (" + y_units + ")"
    if x_units is not None:
        x_label += " (" + x_units + ")"
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_ylim(ymin=0)
    ax.set_xlim(left=0)

    if y_units is '%':
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1,decimals=1))

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, 0.8 * box.width, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ## save FIGURE to png
    figure_dest = lu.get_summary_file_location('figures')
    filename = "paretoplot_all_" + y_data + "_v_" + x_data + ".png"
    plt.savefig(os.path.join(figure_dest, filename))

    if show_plot:
        plt.show()
    else:
        plt.close('all')


def get_data(filename, test_model_dict=None):

    ## get data from CSV
    source = lu.get_summary_file_location('data')
    df_data = pd.read_csv(os.path.join(source,filename), index_col=0)

    if test_model_dict is not None:
        remove_list = []
        for tm,val in test_model_dict.items():
            if not val:
                remove_list.append(tm)

        for rm in remove_list:
            if rm in df_data.index:
                df_data = df_data.drop(rm, axis=0)
            elif rm in df_data.columns:
                df_data = df_data.drop(rm, axis=1)

    return df_data

def generate_pareto_plot(test_case, test_model_dict, y_data='acpf_slack', x_data='solve_time', y_units='p.u', x_units='s',
                         mark_acopf='*', colors=cmap.viridis, annotate_plot=False, show_plot=False):

    ## get data
    _, case_name = os.path.split(test_case)
    case_name, ext = os.path.splitext(case_name)
    input = "summary_data_" + case_name + ".csv"
    df_data = get_data(input, test_model_dict=test_model_dict)

    models = list(df_data.index.values)
    df_y_data = df_data[y_data]
    df_x_data = df_data[x_data]

    ## Create plot
    fig, ax = plt.subplots()

    #---- set property cycles
    marker_style = pareto_marker_style(models,colors=colors)
    for m in models:
        ms = marker_style[m]
        x = df_x_data[m]
        y = df_y_data[m]
        ax.plot(x, y, **ms)

        if annotate_plot:
            ax.annotate(m, (x,y))

    ax.set_title(y_data + " vs. " + x_data + "\n(" + case_name + ")")
    ax.set_ylabel(y_data + " (" + y_units + ")")
    ax.set_xlabel(x_data + " (" + x_units + ")")
    ax.set_xlim(left=0)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, 0.8 * box.width, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ## save FIGURE to png
    figure_dest = lu.get_summary_file_location('figures')
    filename = "paretoplot_" + case_name + "_" + y_data + "_v_" + x_data + ".png"
    plt.savefig(os.path.join(figure_dest, filename))

    if show_plot:
        plt.show()
    else:
        plt.close('all')



def lazy_speedup_plot(test_model_list, colors=None, show_plot=True):

    if colors is None:
        colors=get_colors('inferno')

    lazy_speedup_dict = lu.get_lazy_speedup_dict(test_model_list)
    generate_speedup_data(mean_data='solve_time_geomean', benchmark='acopf')
    generate_speedup_heatmap(test_model_dict=lazy_speedup_dict, mean_data='solve_time_geomean', benchmark='acopf',
                             include_benchmark=False, colormap=colors, cscale='log', show_plot=show_plot)

def trunc_speedup_plot(test_model_list, colors=None, show_plot=True):

    if colors is None:
        colors=get_colors('inferno')

    trunc_speedup_dict = lu.get_trunc_speedup_dict(test_model_list)
    generate_speedup_data(mean_data='solve_time_geomean', benchmark='dlopf_default')
    generate_speedup_heatmap(test_model_dict=trunc_speedup_dict, mean_data='solve_time_geomean', benchmark='dlopf_default',
                             cscale='linear', include_benchmark=True, colormap=colors, show_plot=show_plot)


def draw_graph(G, weighted=False, filename=None, show_plot=True):

    color_dict = {k:None for k in G.nodes}
    for k,c in color_dict.items():
        if k[0:2]=='pf':
            color_dict[k] = 'tab:blue'
        elif k[0:2]=='pl':
            color_dict[k] = 'tab:orange'
        elif k[0:2]=='pg':
            color_dict[k] = 'tab:green'
        elif k[0:2]=='va':
            color_dict[k] = 'tab:red'
        elif k[0:2]=='qf':
            color_dict[k] = 'tab:brown'
        elif k[0:2]=='ql':
            color_dict[k] = 'tab:pink'
        elif k[0:2]=='qg':
            color_dict[k] = 'tab:gray'
        elif k[0:2]=='vm':
            color_dict[k] = 'tab:olive'
        else:
            color_dict[k] = 'tab:purple'

    density = networkx.nx.density(G)
    print('Graph density is {}.'.format(density))

    elist = networkx.to_edgelist(G)
    wlist = [abs(w['weight']) for t,f,w in elist]
    weights = [5 * w / max(wlist) for w in wlist]

    #weights={(u,v):{'weight':1} for u,v in G.edges}
    #networkx.set_edge_attributes(G, weights)
    nlist = create_shell_list(G)

    options = {}
    options['node_color'] = list(color_dict.values())
    options['with_labels'] = False
    options['nlist'] = nlist
    if weighted:
        options['width'] = weights
        options['alpha'] = 0.66
    else:
        options['alpha'] = 0.33
    plt.figure(figsize=(6, 6))
    #networkx.draw_circular(G, **options)
    networkx.draw_shell(G, **options)
    #plt.show()

    ## save FIGURE to png
    if filename is not None:
        figure_dest = lu.get_summary_file_location('figures')
        plt.savefig(os.path.join(figure_dest, filename))

    if show_plot:
        plt.show()
    else:
        plt.close('all')


def create_graph(branch_attrs, bus_attrs):

    G = networkx.Graph()
    G.add_node('syspg')
    G.add_node('syspl')
    G.add_nodes_from(['va'+n for n in bus_attrs['names']])     # voltage angle
    G.add_nodes_from(['pl'+n for n in branch_attrs['names']])  # real loss
    G.add_nodes_from(['pf'+n for n in branch_attrs['names']])  # real flow
    G.add_nodes_from(['pg'+n for n in bus_attrs['names']])     # real injection
    #G.add_node('sysqg')
    G.add_node('sysql')
    G.add_nodes_from(['vm'+n for n in bus_attrs['names']])     # voltage magnitude
    G.add_nodes_from(['ql'+n for n in branch_attrs['names']])  # reactive loss
    G.add_nodes_from(['qf'+n for n in branch_attrs['names']])  # reactive flow
    G.add_nodes_from(['qg'+n for n in bus_attrs['names']])     # reactive injection

    return G

def create_shell_list(G):

    list_sys = ['syspg', 'syspl'] + ['sysql']

    list_gen = [n for n in G.nodes if n[0:2]=='pg']
    list_gen += [n for n in G.nodes if n[0:2]=='qg']

    list_flow = [n for n in G.nodes if n[0:2]=='pf']
    list_flow += [n for n in G.nodes if n[0:2]=='qf']

    list_loss = [n for n in G.nodes if n[0:2]=='pl']
    list_loss += [n for n in G.nodes if n[0:2]=='ql']

    list_volt = [n for n in G.nodes if n[0:2]=='va']
    list_volt += [n for n in G.nodes if n[0:2]=='vm']

    nlist = [list_sys, list_gen, list_loss, list_flow, list_volt]

    return nlist


def remove_unmonitored(G, md, s_tol=None, v_tol=None):
    import math

    num_edges = networkx.number_of_edges(G)
    branches = dict(md.elements(element_type='branch'))
    buses = dict(md.elements(element_type='bus'))

    if s_tol is not None:
        for k,branch in branches.items():
            smax = branch['rating_long_term']
            pf = branch['pf']
            pt = branch['pt']
            qf = branch['qf']
            qt = branch['qt']
            sp = math.sqrt(max(pf**2 + qf**2, pt**2 + qt**2))
            slack = 1 - sp / smax
            if slack > s_tol:
                rm = list(G.edges('pf'+k, data=True))
                G.remove_edges_from(rm)
                rm = list(G.edges('qf'+k, data=True))
                G.remove_edges_from(rm)

    if v_tol is not None:
        for n,bus in buses.items():
            vmax = bus['v_max']
            vmin = bus['v_min']
            vm = bus['vm']
            slack = (vm - vmin) / (vmax - vmin)
            if slack < v_tol or (1-slack) < v_tol:
                rm = list(G.edges('vm'+n, data=True))
                G.remove_edges_from(rm)

    rm_edges = num_edges - networkx.number_of_edges(G)
    print('Removed {} unmonitored edges.'.format(rm_edges))

def remove_truncate(G, md, tol=None):

    num_edges = networkx.number_of_edges(G)
    branches = dict(md.elements(element_type='branch'))
    buses = dict(md.elements(element_type='bus'))

    if tol is not None:
        for k, branch in branches.items():
            maxP = max([abs(df) for df in branch['ptdf'].values()])
            maxQ = max([abs(df) for df in branch['qtdf'].values()])
            remove_p = [('pf'+k, 'pg'+n) for n,df in branch['ptdf'].items() if abs(df)/maxP < tol]
            remove_q = [('qf'+k, 'qg'+n) for n,df in branch['qtdf'].items() if abs(df)/maxQ < tol]
            G.remove_edges_from(remove_p)
            G.remove_edges_from(remove_q)
        for m, bus in buses.items():
            maxV = max([abs(df) for df in bus['vdf'].values()])
            remove_v = [('pg'+n, 'vm'+m) for n,df in bus['vdf'].items() if abs(df)/maxV < tol]
            G.remove_edges_from(remove_v)

    rm_edges = num_edges - networkx.number_of_edges(G)
    print('Removed {} truncated edges.'.format(rm_edges))

def hybridize_losses(G, md, N=None):

    from egret.model_library.transmission.tx_calc import reduce_branches
    branches = dict(md.elements(element_type='branch'))
    active_set = reduce_branches(branches, N)
    num_edges = networkx.number_of_edges(G)

    for k, branch in branches.items():
        if k not in active_set:
            for n, df in branch['pldf'].items():
                rm = list(G.edges('pl'+k, data=True))
                G. remove_edges_from(rm)
                w = df
                if G.has_edge('pg'+n, 'syspg'):
                    w += G.get_edge_data('pg'+n, 'syspg')['weight']
                G.add_edge('pg'+n, 'syspl', weight=abs(w))
            for n, df in branch['qldf'].items():
                rm = list(G.edges('ql'+k, data=True))
                G.remove_edges_from(rm)
                w = df
                if G.has_edge('qg'+n, 'sysqg'):
                    w += G.get_edge_data('qg'+n, 'sysqg')['weight']
                G.add_edge('qg'+n, 'sysql', weight=abs(w))

    rm_edges = num_edges - networkx.number_of_edges(G)
    if rm_edges >= 0:
        print('Removed {} edges with hybrid line losses.'.format(rm_edges))
    else:
        print('Added {} edges for hybrid line losses.'.format(rm_edges))

def plot_sparse_variables(test_case=None, md=None, show_plot=True):

    from egret.models.acopf import solve_acopf

    if test_case is None:
        test_case = lu.idx_to_test_case(0)
        if os.path.basename(os.getcwd()) == 'data':
            test_case = test_case[3:]

    if md is None:
        _md = create_ModelData(test_case)
        md = solve_acopf(_md, solver='ipopt', solver_tee=False)

    #create_dicts_of_lccm(md)
    branches = dict(md.elements(element_type='branch'))
    branch_attrs = md.attributes(element_type='branch')
    bus_attrs = md.attributes(element_type='bus')
    system_attrs = md.data['system']

    # load data into network object
    G = create_graph(branch_attrs, bus_attrs)

    filename = "problem_structure_blank.png"
    draw_graph(G, filename=filename, show_plot=False)

    for k, branch in branches.items():
        for n, w in branch['Ft'].items():
            G.add_edge('pf' + k, 'va' + n, weight=abs(w))
        for n, w in branch['Lt'].items():
            G.add_edge('pl' + k, 'va' + n, weight=abs(w))
        for n, w in branch['Fv'].items():
            G.add_edge('qf'+k, 'vm'+n, weight=abs(w))
        for n, w in branch['Lv'].items():
            G.add_edge('ql'+k, 'vm'+n, weight=abs(w))
        relative_p_loss = 2 * (branch['pf'] + branch['pt']) / abs(branch['pf'] - branch['pt'])
        relative_q_loss = 2 * (branch['qf'] + branch['qt']) / abs(branch['qf'] - branch['qt'])
        G.add_edge('pf'+k, 'pl'+k, weight=relative_p_loss)
        G.add_edge('qf'+k, 'ql'+k, weight=abs(relative_q_loss))

    Ad = scipy.sparse.coo_matrix(system_attrs['AdjacencyMat'])
    for i,j,v in zip(Ad.row, Ad.col, Ad.data):
        pg = 'pg'+str(i+1)
        qg = 'qg'+str(i+1)
        pf = 'pf'+str(j+1)
        qf = 'qf'+str(j+1)
        pl = 'pl'+str(j+1)
        ql = 'ql'+str(j+1)
        G.add_edge(pg, pf, weight=abs(v))
        G.add_edge(pg, pl, weight=0.5*abs(v))
        G.add_edge(qg, qf, weight=abs(v))
        G.add_edge(qg, ql, weight=0.5*abs(v))

    filename = "problem_structure_sparse.png"
    draw_graph(G, filename=filename, show_plot=True)


def plot_dense_variables(test_case=None, md=None, show_plot=True):

    from egret.models.acopf import solve_acopf

    if test_case is None:
        test_case = lu.idx_to_test_case(0)
        if os.path.basename(os.getcwd()) == 'data':
            test_case = test_case[3:]

    if md is None:
        _md = create_ModelData(test_case)
        md = solve_acopf(_md, solver='ipopt', solver_tee=False)

    #create_dicts_of_fdf(md)
    branches = dict(md.elements(element_type='branch'))
    buses = dict(md.elements(element_type='bus'))
    branch_attrs = md.attributes(element_type='branch')
    bus_attrs = md.attributes(element_type='bus')
    system_attrs = md.data['system']

    # load data into network object
    G = create_graph(branch_attrs, bus_attrs)

    for k, branch in branches.items():
        for n, w in branch['ptdf'].items():
            G.add_edge('pf' + k, 'pg' + n, weight=abs(w))
        for n, w in branch['pldf'].items():
            G.add_edge('pl' + k, 'pg' + n, weight=abs(w))
        for n, w in branch['qtdf'].items():
            G.add_edge('qf'+k, 'qg'+n, weight=abs(w))
        for n, w in branch['qldf'].items():
            G.add_edge('ql'+k, 'qg'+n, weight=abs(w))
        relative_p_loss = 2 * (branch['pf'] + branch['pt']) / abs(branch['pf'] - branch['pt'])
        relative_q_loss = 2 * (branch['qf'] + branch['qt']) / abs(branch['qf'] - branch['qt'])
        G.add_edge('pf'+k, 'pl'+k, weight=relative_p_loss)
        G.add_edge('qf'+k, 'ql'+k, weight=abs(relative_q_loss))
        G.add_edge('pl'+k, 'syspg', weight=1)
        #G.add_edge('ql'+k, 'sysqg', weight=1)

    for n, bus in buses.items():
        for m, w in bus['vdf'].items():
            G.add_edge('vm'+n, 'qg'+m, weight=abs(w))
        G.add_edge('pg'+n, 'syspg', weight=1)
        #G.add_edge('qg'+n, 'sysqg', weight=1)

    filename = "problem_structure_dense_full.png"
    draw_graph(G, show_plot=False, filename=filename)

    _len_bus = len(buses.keys())
    _len_branch = len(branches.keys())
    _len_cycle = _len_branch - _len_bus + 1
    remove_unmonitored(G, md, s_tol=0.5, v_tol=0.25)
    hybridize_losses(G, md, N=_len_cycle)
    remove_truncate(G, md, tol=1e-2)

    filename = "problem_structure_dense.png"
    draw_graph(G, show_plot=show_plot, filename=filename)


def plot_compact_variables(test_case=None, md=None, show_plot=True):

    from egret.models.acopf import solve_acopf

    if test_case is None:
        test_case = lu.idx_to_test_case(0)
        if os.path.basename(os.getcwd()) == 'data':
            test_case = test_case[3:]

    if md is None:
        _md = create_ModelData(test_case)
        md = solve_acopf(_md, solver='ipopt', solver_tee=False)

    #create_dicts_of_fdf(md)
    branches = dict(md.elements(element_type='branch'))
    buses = dict(md.elements(element_type='bus'))
    branch_attrs = md.attributes(element_type='branch')
    bus_attrs = md.attributes(element_type='bus')
    system_attrs = md.data['system']

    # load data into network object
    G = create_graph(branch_attrs, bus_attrs)

    for k, branch in branches.items():
        for n, w in branch['ptdf'].items():
            G.add_edge('pf' + k, 'pg' + n, weight=abs(w))
        for n, w in branch['qtdf'].items():
            G.add_edge('qf'+k, 'qg'+n, weight=abs(w))
        G.add_edge('pf'+k, 'syspl', weight=branch['ploss_distribution'])
        G.add_edge('qf'+k, 'sysql', weight=abs(branch['qloss_distribution']))

    for n, bus in buses.items():
        G.add_edge('pg'+n, 'syspl', weight=abs(bus['ploss_sens']))
        G.add_edge('qg'+n, 'sysql', weight=abs(bus['qloss_sens']))
        G.add_edge('pg'+n, 'syspg', weight=1)
        #G.add_edge('qg'+n, 'sysqg', weight=1)
        for m, w in bus['vdf'].items():
            G.add_edge('qg'+n, 'vm'+m, weight=abs(w))

    G.add_edge('syspg', 'syspl', weight=1)
    #G.add_edge('sysqg', 'sysql', weight=1)

    filename = "problem_structure_compact_full.png"
    draw_graph(G, filename=filename, show_plot=False)

    remove_unmonitored(G, md, s_tol=0.5, v_tol=0.25)
    remove_truncate(G, md, tol=1e-2)

    filename = "problem_structure_compact.png"
    draw_graph(G, filename=filename, show_plot=show_plot)


def plot_ptdf_variables(test_case=None, md=None, show_plot=True):

    from egret.models.acopf import solve_acopf

    if test_case is None:
        test_case = lu.idx_to_test_case(0)
        if os.path.basename(os.getcwd()) == 'data':
            test_case = test_case[3:]

    if md is None:
        _md = create_ModelData(test_case)
        md = solve_acopf(_md, solver='ipopt', solver_tee=False)

    #create_dicts_of_fdf(md)
    branches = dict(md.elements(element_type='branch'))
    buses = dict(md.elements(element_type='bus'))
    branch_attrs = md.attributes(element_type='branch')
    bus_attrs = md.attributes(element_type='bus')
    system_attrs = md.data['system']

    # load data into network object
    G = create_graph(branch_attrs, bus_attrs)

    for k, branch in branches.items():
        for n, w in branch['ptdf'].items():
            G.add_edge('pf' + k, 'pg' + n, weight=abs(w))
        G.add_edge('pf'+k, 'syspl', weight=abs(branch['ploss_distribution']))

    for n, bus in buses.items():
        G.add_edge('pg'+n, 'syspl', weight=abs(bus['ploss_sens']))
        G.add_edge('pg'+n, 'syspg', weight=1)

    G.add_edge('syspg', 'syspl', weight=1)

    filename = "problem_structure_ptdf_full.png"
    draw_graph(G, show_plot=False, filename=filename)

    remove_unmonitored(G, md, s_tol=0.5)
    remove_truncate(G, md, tol=1e-2)

    filename = "problem_structure_ptdf.png"
    draw_graph(G, show_plot=show_plot, filename=filename)


def plot_constraint_matrices(test_case=None):

    if test_case == None:
        test_case = test.get_case_names(flag='pglib_opf_case14_ieee')[0]
    test_case = lu.idx_to_test_case(test_case)
    if os.path.basename(os.getcwd()) == 'data':
        test_case = test_case[3:]

    plot_sparse_variables(test_case)
    plot_dense_variables(test_case)
    plot_compact_variables(test_case)
    plot_ptdf_variables(test_case)


def generate_plots(show_plot=False):

    # Generate plots
    case_dict = test.get_case_dict()
    #generate_plot_data(case_dict['ieee'])
    #generate_plot_data(case_dict['k'])
    generate_boxplots(show_plot=show_plot)
    generate_barplots(case_dict['ieee'], list_name='ieee', show_plot=show_plot)
    generate_barplots(case_dict['k'], list_name='polish', show_plot=show_plot)
    generate_heatmaps(case_dict['ieee'], show_plot=show_plot)
    generate_heatmaps(case_dict['k'], show_plot=show_plot)
    generate_sensitivities(case_dict['ieee'], show_plot=show_plot)
    summarize_sensitivities(flag='ieee',show_plot=show_plot)

def create_full_summary(show_plot=False):

    # Generate data files
    case_sets = ['ieee','k','rte','sdet','tamu','pegase','misc']
    case_dict = test.get_case_dict()
    for k,case_list in case_dict.items():
        update_case_data(case_list, tag=k)
        update_data_tables(tag=k)
    update_all_case_data(case_sets)
    nominal_all_data(case_sets, 'solve_time')

    # Generate plots
    generate_plots(show_plot=show_plot)
    plot_constraint_matrices()


if __name__ == '__main__':

    create_full_summary(show_plot=False)
    #summarize_sensitivities(flag='ieee',show_plot=True)
    #generate_boxplots(show_plot=True)
    #generate_plots(show_plot=True)
    #summarize_sensitivities(flag='ieee', show_plot=True)
    #generate_boxplots(show_plot=True)
    #plot_constraint_matrices()
