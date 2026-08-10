from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test
from lifelines.statistics import pairwise_logrank_test
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns

def train_cox(data_dict, fold,  penalizer, l1_ratio, event_data_col, event_ind_col, cols_to_include, additional_flag=True, robust=True, save_path=None):
    train, test = data_dict[fold]['data_all']['train'], data_dict[fold]['data_all']['test']
    if additional_flag:
        additional = data_dict[fold]['data_all']['additional']

    cols_to_include_ = cols_to_include + [event_data_col, event_ind_col]

    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)

    try:
        cph.fit(train[cols_to_include_], event_data_col, event_ind_col, show_progress=False)
    except ValueError:
        train = train[~train[cols_to_include_].isna()]
        cph.fit(train, event_data_col, event_ind_col, show_progress=False, robust=robust)

    summary = cph.summary
    summary = summary.sort_values(by='coef', ascending=True)

    predictions = dict()
    predictions['train'] = cph.predict_partial_hazard(train)
    predictions['test'] = cph.predict_partial_hazard(test)
    predictions['additional'] = cph.predict_partial_hazard(additional)

    train_ci = concordance_index(train[event_data_col], -cph.predict_partial_hazard(train), train[event_ind_col])
    test_ci = concordance_index(test[event_data_col], -cph.predict_partial_hazard(test), test[event_ind_col])
    if additional_flag:
        additional_ci = concordance_index(additional[event_data_col], -cph.predict_partial_hazard(additional), additional[event_ind_col])
    else:
        additional_ci = 0

    if save_path is not None:
        summary.to_csv(os.path.join(save_path, f'{event_ind_col}_fold_{fold}.csv'))
    print(f'Fold {fold} Alpha {penalizer:.3f}\tC-index (Train/test/additional): {train_ci:.2f}/{test_ci:.2f}/{additional_ci:.2f}')

    return summary, [train_ci, test_ci, additional_ci], predictions, cph

def get_best_alpha(alpha_path):
    try:
        results_path = os.path.join(alpha_path, 'c_indexes.csv')
        results = pd.read_csv(results_path)
    except:
        results = alpha_path

    alphas = np.unique(results['alpha'].values)
    mean_cis = list()

    for alpha in alphas:
        subset = results[results['alpha'] == alpha]
        train = subset['train'].mean()
        test = subset['test'].mean()
        additional = subset['additional'].mean()
        if additional > 0:
            # mean_cis.append((train, test, additional))
            mean_cis.append(np.mean((train, test, additional)))
        else:
            mean_cis.append((train, test))

    max_ci = max(mean_cis)
    alpha_index = mean_cis.index(max_ci)

    best_alpha = alphas[alpha_index]
    
    return best_alpha

def plot_km_two_groups(df, event_ind_field, event_data_field, group_col, max_months, add_counts, ci_show, title, ax):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,8))
    time = np.linspace(0, max_months)
    high_colour = 'tab:blue'
    low_colour = 'tab:orange'
    kmf_l = KaplanMeierFitter(label='Low Risk')
    kmf_l.fit(df[df[group_col] == 0][event_data_field], df[df[group_col] == 0][event_ind_field], timeline=time)
    kmf_h = KaplanMeierFitter(label='High Risk')
    kmf_h.fit(df[df[group_col] == 1][event_data_field], df[df[group_col] == 1][event_ind_field], timeline=time)
    kmf_l.plot_survival_function(show_censors=True, ci_show=ci_show, ax=ax, lw=2, color=low_colour)
    kmf_h.plot_survival_function(show_censors=True, ci_show=ci_show, ax=ax, lw=2, color=high_colour)

    if add_counts:
        # add_at_risk_counts(kmf_l, kmf_h, rows_to_show=['At risk'], ax=ax)
        add_at_risk_counts(kmf_l, kmf_h, ax=ax)

    result = logrank_test(df[df[group_col] == 1][event_data_field].values, df[df[group_col] == 0][event_data_field].values, df[df[group_col] == 1][event_ind_field].values, df[df[group_col] == 0][event_ind_field].values)
    p_val = np.round(result.p_value, 3)
    if p_val < 0.001:
        p_val = 'p < 0.001'
    else:
        p_val = f'p = {p_val}'

    ax.set_title(f'{title}')
    ax.text(x=3, y=0.05, s=f'Log rank test \n {p_val}')
    ax.set_ylim([0.1,1.10])
    ax.set_ylabel('Survival Probability\n (Overall Survival)')
    ax.set_xlabel('Timeline (Months)')
    ax.set_xticks(ticks=ax.get_xticks())
    ax.set_yticks(ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend()
    if max_months is None:
        max_months = df[event_data_field].values.max() + 6
    ax.set_xlim([0.0, max_months])
    plt.tight_layout()
    print(p_val)

def merge_clinical(data, survival, additional_survival, merge_cols):
    datas = list()
    for j, set_name in enumerate([data[i][1] for i in range(4)]): # no grade data for TCGA
        set_data = data[j][0]
        if set_name == 'additional':
            # datas.append((set_data, set_name))
            set_data = set_data.merge(additional_survival[['samples']+[merge_cols]], on='samples')
        else:
            set_data = set_data.merge(survival[['samples']+[merge_cols]], on='samples')
        datas.append((set_data, set_name))
    return datas

def cross_val_survival(datas, event_ind_field, event_data_field, cols_to_include, fold, alphas):
    fold_data = datas[fold]
    train, test, additional = fold_data['data']['train'], fold_data['data']['test'], fold_data['data']['additional'] 
    c_indexes = list()

    for alpha in alphas:
        _, ci, _, _ = train_cox(data_dict=datas, fold=fold, penalizer=alpha, l1_ratio=0.0, event_ind_col=event_ind_field, event_data_col=event_data_field, cols_to_include=cols_to_include, additional_flag=True, save_path=None)
        c_indexes.append(ci+[alpha])

    cis_df = pd.DataFrame(c_indexes, columns=['train', 'test', 'additional', 'alpha'])
    cis_df = cis_df.reset_index()
    cis_df.rename(columns={'index':'fold'}, inplace=True)
    # c_indexes.append(cis_df)
    
    # c_indexes_df = pd.concat(c_indexes)
    best_alpha = get_best_alpha(cis_df)

    _, ci, _, _ = train_cox(data_dict=datas, fold=fold, penalizer=best_alpha, l1_ratio=0.0, event_ind_col=event_ind_field, event_data_col=event_data_field, cols_to_include=cols_to_include, additional_flag=True, save_path=None)
    return ci+[best_alpha]

def get_high_low_risks(data_dict, predictions, fold, alpha_path, q_buckets=2):
    fold_data_all = data_dict[fold]['data_all']
    high_low = list()
    for set_name in fold_data_all.keys():
        current_predictions = predictions[set_name].copy(deep=True)
        current_data = fold_data_all[set_name].copy(deep=True)
        current_data['hazard'] = current_predictions

        if set_name == 'train':
            if q_buckets == 2:
                hazard_cutoff = np.median(current_predictions)
            else:
                cutoffs = create_risk_buckets(predictions=predictions, num_groups=q_buckets)
        
        if q_buckets == 2:
            current_data[f'h_bin_{fold}'] = (current_data['hazard'] > hazard_cutoff) * 1
        else:
            current_data[f'h_bin_{fold}'] = pd.cut(current_data['hazard'], bins=[-np.inf] + list(cutoffs) + [np.inf],
                                                   labels=[i for i in range(len(list(cutoffs))+1)])

        current_data['original_set'] = set_name
        high_low.append(current_data)
    
    high_low_frame = pd.concat(high_low)
    high_low_frame['fold'] = fold
    high_low_frame.to_csv(os.path.join(alpha_path, f'high_low_fold_{fold}.csv'))
    return high_low_frame

def create_risk_buckets(predictions, num_groups):
    sorted_predictions = sorted(predictions['train'])
    percentiles = [i * 100 / num_groups for i in range(1, num_groups)]
    cutoffs = np.percentile(sorted_predictions, percentiles)
    print(cutoffs)
    return cutoffs

def plot_kaplan_meier(
    df, 
    duration_col, 
    event_col, 
    group_col, 
    ax=None, 
    title="Kaplan-Meier Survival Curve", 
    xlabel="Time", 
    ylabel="Survival Probability",
    ci_show=True,
    at_risk_counts=True,
    plot_kwargs=None,
    risk_table_kwargs=None
):
    """
    Generate a Kaplan-Meier survival curve for multiple groups.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the survival data.
    duration_col : str
        Column name for the duration/time values.
    event_col : str
        Column name for the event indicator (1 for event, 0 for censored).
    group_col : str
        Column name for the group indicator.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure and axes will be created.
    title : str, optional
        Title for the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    ci_show : bool, optional
        Whether to show confidence intervals.
    at_risk_counts : bool, optional
        Whether to display the number at risk at the bottom of the plot.
    plot_kwargs : dict, optional
        Additional keyword arguments to pass to the plot method.
    risk_table_kwargs : dict, optional
        Additional keyword arguments to pass to the at_risk_counts method.
        
    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes with the Kaplan-Meier plot.
    kmfs : dict
        Dictionary of KaplanMeierFitter objects for each group.
    """
    # Default plot kwargs
    if plot_kwargs is None:
        plot_kwargs = {}
    
    # Default risk table kwargs
    if risk_table_kwargs is None:
        risk_table_kwargs = {}
    
    # Create new figure and axes if not provided
    if ax is None:
        if at_risk_counts:
            fig, ax = plt.subplots(nrows=2, ncols=1, 
                                  gridspec_kw={'height_ratios': [3, 1]}, 
                                  figsize=(10, 8), sharex=True)
            risk_ax = ax[1]
            ax = ax[0]
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            risk_ax = None
    else:
        risk_ax = None
        if at_risk_counts:
            print("Warning: at_risk_counts is True but only one ax provided. "
                  "Risk table will not be shown.")
            at_risk_counts = False
    
    # Get unique groups
    groups = df[group_col].unique()
    
    # Fit Kaplan-Meier for each group
    kmfs = {}
    colors = plt.cm.Set2.colors[:len(groups)]
    
    # Create a label for the legend
    legend_elements = []
    
    # If we're showing the at-risk counts table, initialize the times to use
    if at_risk_counts:
        timeline = np.linspace(0, df[duration_col].max(), num=10)
        risk_groups = {str(group): [] for group in groups}
    
    # Loop through each group and fit KM curve
    for i, group in enumerate(groups):
        group_df = df[df[group_col] == group]
        
        kmf = KaplanMeierFitter()
        label = f"{group}"
        # label = f"{group} (n={len(group_df)})"
        
        kmf.fit(
            group_df[duration_col], 
            group_df[event_col], 
            label=label
        )
        
        # Store the fitter for later use
        kmfs[group] = kmf
        
        # Default color cycling if not specified
        color = colors[i % len(colors)]
        if 'color' not in plot_kwargs:
            current_plot_kwargs = {**plot_kwargs, 'color': color}
        else:
            current_plot_kwargs = plot_kwargs
        
        # Plot the KM curve
        kmf.plot_survival_function(
            ax=ax, 
            ci_show=ci_show,
            show_censors=True,
            **current_plot_kwargs
        )
        
        # Add to legend
        legend_elements.append(
            Line2D([0], [0], color=color, lw=3, label=label)
        )
        
        # If we're showing the at-risk counts, calculate them
        if at_risk_counts:
            risk_data = kmf.survival_function_at_times(timeline)
            counts = []
            for t in timeline:
                count = (group_df[duration_col] >= t).sum()
                counts.append(count)
            risk_groups[str(group)] = counts
    
    # Add the at-risk table if requested
    if at_risk_counts and risk_ax is not None:
        # Hide the risk_ax frame
        risk_ax.set_frame_on(False)
        
        # Create header row
        risk_ax.text(-0.1, 1.1, "Time", ha='center', va='center', fontweight='bold')
        risk_ax.text(-0.1, 0.5, "At risk", ha='center', va='center', fontweight='bold')
        
        # Add times on top
        for i, t in enumerate(timeline):
            risk_ax.text(i / (len(timeline) - 1), 1.1, f"{t:.0f}", ha='center', va='center')
        
        # Add risk counts for each group
        for j, (group, counts) in enumerate(risk_groups.items()):
            # Position the group name to the left
            risk_ax.text(-0.2, 0.5 - 0.5*j/len(risk_groups), group, ha='left', va='center')
            
            # Add counts for each time point
            for i, count in enumerate(counts):
                risk_ax.text(i / (len(timeline) - 1), 0.5 - 0.5*j/len(risk_groups), 
                            str(count), ha='center', va='center')
        
        # Turn off axis ticks and labels
        risk_ax.set_xticks([])
        risk_ax.set_yticks([])
        
        # Set risk table height
        risk_ax.set_ylim(-0.5, 1.5)
    
    # Set plot aesthetics
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.grid(True, alpha=0.3)
    
    # Add legend with custom elements
    ax.legend(handles=legend_elements, loc='best')
    
    # Set ylim to start from 0
    ax.set_ylim(0, 1.05)
    
    # Return the axes and KMF objects
    return ax, kmfs