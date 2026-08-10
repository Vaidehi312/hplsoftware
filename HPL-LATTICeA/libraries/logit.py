from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import RandomOverSampler
from matplotlib.font_manager import FontProperties
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
from sklearn.metrics import *
from sklearn.model_selection import StratifiedKFold
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore", message="FutureWarning")

import sys
sys.path.append('/nfs/home/users/krakovic/sharedscratch/Histomorphological-Phenotype-Learning/')
from models.clustering.logistic_regression_leiden_clusters import *

sys.path.append('/nfs/home/users/krakovic/sharedscratch/notebooks/latticea_he/libraries')
from clinical import load_clinical
from data_processing import *


def run_logit(data_dict, fold, leiden_clusters, alpha):
	print(f'\tFold: {fold} (Train/test/additional)')
	train_data = data_dict[fold]['train_X']
	train_labels = data_dict[fold]['train_y']

	model = sm.Logit(endog=train_labels, exog=train_data).fit_regularized(method='l1', alpha=alpha, disp=0)

	test_data = data_dict[fold]['test_X']
	test_labels = data_dict[fold]['test_y']

	try:
		additional_data = data_dict[fold]['additional_X']
		additional_labels = data_dict[fold]['additional_y']
	except KeyError:
		additional_data = None
		additional_labels = None

	train_pred = model.predict(exog=train_data)
	test_pred = model.predict(exog=test_data)
	additional_pred = model.predict(exog=additional_data)

	train_pred_labs = (train_pred >= 0.5).astype(int) 
	test_pred_labs = (test_pred >= 0.5).astype(int)
	additional_pred_labs = (additional_pred >= 0.5).astype(int)

	train_auc = roc_auc_score(y_true=train_labels, y_score=train_pred)
	test_auc = roc_auc_score(y_true=test_labels, y_score=test_pred)
	train_acc = accuracy_score(y_true=train_labels, y_pred=train_pred_labs)
	test_acc = accuracy_score(y_true=test_labels, y_pred=test_pred_labs)
	train_f1 = f1_score(y_true=train_labels, y_pred=train_pred_labs)
	test_f1 = f1_score(y_true=test_labels, y_pred=test_pred_labs)
	train_prec = precision_score(y_true=train_labels, y_pred=train_pred_labs)
	test_prec = precision_score(y_true=test_labels, y_pred=test_pred_labs)
	train_rec = recall_score(y_true=train_labels, y_pred=train_pred_labs)
	test_rec = recall_score(y_true=test_labels, y_pred=test_pred_labs)

	if additional_data is not None:
		additional_auc = roc_auc_score(y_true=additional_labels, y_score=additional_pred)
		additional_acc = accuracy_score(y_true=additional_labels, y_pred=additional_pred_labs)
		additional_f1 = f1_score(y_true=additional_labels, y_pred=additional_pred_labs)
		additional_prec = precision_score(y_true=additional_labels, y_pred=additional_pred_labs)
		additional_rec = recall_score(y_true=additional_labels, y_pred=additional_pred_labs)
	else:
		additional_auc = 0
		additional_acc = 0
		additional_f1 = 0
		additional_prec = 0
		additional_rec = 0

	print(f'\t\tAUC: {train_auc:.2f}/{test_auc:.2f}/{additional_auc:.2f}')
	print(f'\t\tAccuracy: {train_acc:.2f}/{test_acc:.2f}/{additional_acc:.2f}')
	print(f'\t\tF1 Score: {train_f1:.2f}/{test_f1:.2f}/{additional_f1:.2f}')
	print(f'\t\tPrecision: {train_prec:.2f}/{test_prec:.2f}/{additional_prec:.2f}')
	print(f'\t\tRecall: {train_rec:.2f}/{test_rec:.2f}/{additional_rec:.2f}')

	scores_df = pd.DataFrame(data=[[fold, 'train', alpha, train_auc, train_acc, train_f1, train_prec, train_rec],
								   [fold, 'test', alpha, test_auc, test_acc, test_f1, test_prec, test_rec],
								   [fold, 'additional', alpha, additional_auc, additional_acc, additional_f1, additional_prec, additional_rec]], 
								   columns=['fold', 'set', 'alpha', 'auc', 'accuracy', 'f1_score', 'precision', 'recall'])

	results_summary = model.summary()
	results_as_html = results_summary.tables[1].as_html()
	results_df      = pd.read_html(results_as_html, header=0)[0]
	results_df['hpc'] = leiden_clusters
	# results_df = results_df.set_index('hpc')
	results_df['exp_coef'] = np.exp(results_df['coef'])
	results_df['exp_0.025'] = np.exp(results_df['[0.025'])
	results_df['exp_0.975'] = np.exp(results_df['0.975]'])

	# results_df = results_df[['hpc', 'coef', 'exp_coef', 'P>|z|', 'exp_0.025', 'exp_0.975', '[0.025', '0.975]']]

	return results_df, scores_df

def forest_plots_logit(results_df, scale_to_plot, results_dir, csv_name):
    labels = results_df['hpc'].values.tolist()
    if scale_to_plot == 'exp':
        effect_measure = np.round(results_df['exp_coef'], 3).values.tolist()
        lower = np.round(results_df['exp_0.025'], 3).values.tolist()
        upper = np.round(results_df['exp_0.975'], 3).values.tolist()
        p_vals = np.round(results_df['P>|z|'], 3).values.tolist()
        center = 1
        effect_measure_lab = 'Odds Ratio'
        xlim = [0, 2]
        xticks = [0, 0.5, 1.0, 1.5, 2.0]
    elif scale_to_plot == 'log':
        effect_measure = np.round(results_df['coef'], 3).values.tolist()
        lower = np.round(results_df['[0.025'], 3).values.tolist()
        upper = np.round(results_df['0.975]'], 3).values.tolist()
        p_vals = np.round(results_df['P>|z|'], 3).values.tolist()
        center = 0
        effect_measure_lab = 'Log Odds Ratio'
        xlim = [-1, 1]
        xticks = [-1, -0.5, 0, 0.5, 1.0]
    else:
        raise Exception("Specify either 'log' or 'exp' scale")

    plot = EffectMeasurePlot_LR(label=labels, effect_measure=effect_measure, lcl=lower, ucl=upper, pvalues=p_vals, center=center)
    plot.labels(effectmeasure=effect_measure_lab)
    plot.colors(pointshape="o")

    ax = plot.plot(figsize=(13,20), size=8, t_adjuster=0.01, max_value=xlim[1], min_value=xlim[0], p_th=0.05)
    plt.suptitle("Cluster",x=0.11,y=0.89, fontweight='bold')
    ax.set_xlabel(effect_measure_lab, x=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    ax.set_xlim(xlim)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    fig_name = csv_name.replace('.csv', '.jpg')
    fname = os.path.join(results_dir, fig_name)
    
    plt.gcf()
    plt.savefig(fname=fname, bbox_inches='tight')
    plt.close()

    # return ax
	
class EffectMeasurePlot_LR:

    """
    The following functions (and their purposes) live within effectmeasure_plot
    labels(**kwargs)
        Used to change the labels in the plot, as well as the center and scale. Inputs are
        keyword arguments
        KEYWORDS:
            -effectmeasure  + changes the effect measure label
            -conf_int       + changes the confidence interval label
            -scale          + changes the scale to either log or linear
            -center         + changes the reference line for the center
    colors(**kwargs)
        Used to change the color of points and lines. Also can change the shape of points.
        Valid colors and shapes for matplotlib are required. Inputs are keyword arguments
        KEYWORDS:
            -errorbarcolor  + changes the error bar colors
            -linecolor      + changes the color of the reference line
            -pointcolor     + changes the color of the points
            -pointshape     + changes the shape of points
    plot(t_adjuster=0.01,decimal=3,size=3)
        Generates the effect measure plot of the input lists according to the pre-specified
        colors, shapes, and labels of the class object
        Arguments:
            -t_adjuster     + used to refine alignment of the table with the line graphs.
                              When generate plots, trial and error for this value are usually
                              necessary
            -decimal        + number of decimal places to display in the table
            -size           + size of the plot to generate
    """

    def __init__(self, label, effect_measure, lcl, ucl, pvalues, center=0):
        """Initializes effectmeasure_plot with desired data to plot. All lists should be the same
        length. If a blank space is desired in the plot, add an empty character object (' ') to
        each list at the desired point.
        Inputs:
        label
            -list of labels to use for y-axis
        effect_measure
            -list of numbers for point estimates to plot. If point estimate has trailing zeroes,
             input as a character object rather than a float
        lcl
            -list of numbers for upper confidence limits to plot. If point estimate has trailing
             zeroes, input as a character object rather than a float
        ucl
            -list of numbers for upper confidence limits to plot. If point estimate has
             trailing zeroes, input as a character object rather than a float
        """
        self.df = pd.DataFrame()
        self.df['study'] = label
        self.df['OR']    = effect_measure
        self.df['LCL']   = lcl
        self.df['UCL']   = ucl
        self.df['P']     = pvalues
        # self.df['S']     = subtypes
        # self.df['Pu']    = purities
        # self.df['C']     = counts
        # self.df['M']     = mean_tp
        # self.df['Ma']    = max_tp
        # self.df['Pp']    = perc_pat
        self.df['OR2']   = self.df['OR'].astype(str).astype(float)
        if (all(isinstance(item, float) for item in lcl)) & (all(isinstance(item, float) for item in effect_measure)):
            self.df['LCL_dif'] = self.df['OR'] - self.df['LCL']
        else:
            self.df['LCL_dif'] = (pd.to_numeric(self.df['OR'])) - (pd.to_numeric(self.df['LCL']))
        if (all(isinstance(item, float) for item in ucl)) & (all(isinstance(item, float) for item in effect_measure)):
            self.df['UCL_dif'] = self.df['UCL'] - self.df['OR']
        else:
            self.df['UCL_dif'] = (pd.to_numeric(self.df['UCL'])) - (pd.to_numeric(self.df['OR']))
        self.em       = 'OR'
        self.ci       = '95% CI'
        self.p        = 'P-Value'
        self.subtype  = 'Subtype'
        self.purity   = 'Purity %'
        self.counts   = 'Tile Counts'
        self.mean_tp  = 'Mean Tile Pat.'
        self.max_tp   = 'Max Tile Pat. %'
        self.perc_pat = 'Patients %'
        self.scale    = 'linear'
        self.center   = center
        self.errc     = 'dimgrey'
        self.shape    = 'o'
        self.pc       = 'k'
        self.linec    = 'gray'

    def labels(self, **kwargs):
        """Function to change the labels of the outputted table. Additionally, the scale and reference
        value can be changed.
        Accepts the following keyword arguments:
        effectmeasure
            -changes the effect measure label
        conf_int
            -changes the confidence interval label
        scale
            -changes the scale to either log or linear
        center
            -changes the reference line for the center
        """
        if 'effectmeasure' in kwargs:
            self.em = kwargs['effectmeasure']
        if 'ci' in kwargs:
            self.ci = kwargs['conf_int']
        if 'scale' in kwargs:
            self.scale = kwargs['scale']
        if 'center' in kwargs:
            self.center = kwargs['center']

    def colors(self, **kwargs):
        """Function to change colors and shapes.
        Accepts the following keyword arguments:
        errorbarcolor
            -changes the error bar colors
        linecolor
            -changes the color of the reference line
        pointcolor
            -changes the color of the points
        pointshape
            -changes the shape of points
        """
        if 'errorbarcolor' in kwargs:
            self.errc = kwargs['errorbarcolor']
        if 'pointshape' in kwargs:
            self.shape = kwargs['pointshape']
        if 'linecolor' in kwargs:
            self.linec = kwargs['linecolor']
        if 'pointcolor' in kwargs:
            self.pc = kwargs['pointcolor']

    def plot(self, figsize=(3, 3), t_adjuster=0.01, decimal=6, size=3, max_value=None, min_value=None, fontsize=None, p_th=0.05):
        """Generates the matplotlib effect measure plot with the default or specified attributes.
        The following variables can be used to further fine-tune the effect measure plot
        t_adjuster
            -used to refine alignment of the table with the line graphs. When generate plots, trial
             and error for this value are usually necessary. I haven't come up with an algorithm to
             determine this yet...
        decimal
            -number of decimal places to display in the table
        size
            -size of the plot to generate
        max_value
            -maximum value of x-axis scale. Default is None, which automatically determines max value
        min_value
            -minimum value of x-axis scale. Default is None, which automatically determines min value
        """
        tval = []
        ytick = []
        for i in range(len(self.df)):
            if (np.isnan(self.df['OR2'][i]) == False):
                if ((isinstance(self.df['OR'][i], float)) & (isinstance(self.df['LCL'][i], float)) & (isinstance(self.df['UCL'][i], float))):
                    list_val = [round(self.df['OR2'][i], decimal), ('(' + str(round(self.df['LCL'][i], decimal)) + ', ' + str(round(self.df['UCL'][i], decimal)) + ')'), str(self.df['P'][i])]
                    tval.append(list_val)
                else:
                    list_val = [self.df['OR'][i], ('(' + str(self.df['LCL'][i]) + ', ' + str(self.df['UCL'][i]) + ')'), self.df['P'][i], self.df['S'][i], self.df['Pu'][i], self.df['C'][i], \
                                self.df['M'][i], self.df['Ma'][i], self.df['Pp'][i]]
                    tval.append()
                ytick.append(i)
            else:
                tval.append([' ', ' ', ' ', ' '])
                ytick.append(i)
        if max_value is None:
            if pd.to_numeric(self.df['UCL']).max() < 1:
                maxi = round(((pd.to_numeric(self.df['UCL'])).max() + 0.05),
                             6)  # setting x-axis maximum for UCL less than 1
            if (pd.to_numeric(self.df['UCL']).max() < 9) and (pd.to_numeric(self.df['UCL']).max() >= 1):
                maxi = round(((pd.to_numeric(self.df['UCL'])).max() + 1),
                             0)  # setting x-axis maximum for UCL less than 10
            if pd.to_numeric(self.df['UCL']).max() > 9:
                maxi = round(((pd.to_numeric(self.df['UCL'])).max() + 10),
                             0)  # setting x-axis maximum for UCL less than 100
        else:
            maxi = max_value
        if min_value is None:
            if pd.to_numeric(self.df['LCL']).min() > 0:
                mini = round(((pd.to_numeric(self.df['LCL'])).min() - 0.1), 6)  # setting x-axis minimum
            if pd.to_numeric(self.df['LCL']).min() < 0:
                mini = round(((pd.to_numeric(self.df['LCL'])).min() - 0.05), 6)  # setting x-axis minimum
        else:
            mini = min_value
        plt.figure(figsize=figsize)  # blank figure
        gspec = gridspec.GridSpec(1, 4)  # sets up grid
        plot = plt.subplot(gspec[0, 0:3])  # plot of data
        tabl = plt.subplot(gspec[0, 3:])  # table of OR & CI
        plot.set_ylim(-1, (len(self.df)))  # spacing out y-axis properly
        if self.scale == 'log':
            try:
                plot.set_xscale('log')
            except:
                raise ValueError('For the log scale, all values must be positive')
        plot.axvline(self.center, color=self.linec, zorder=1)
        plot.errorbar(self.df.OR2, self.df.index, xerr=[self.df.LCL_dif, self.df.UCL_dif], marker='None', zorder=2, ecolor=self.errc, elinewidth=(size*3 / size), linewidth=0)
        plot.scatter(self.df.OR2, self.df.index, c=self.pc, s=(size * 25), marker=self.shape, zorder=3, edgecolors='None')
        plot.xaxis.set_ticks_position('bottom')
        plot.yaxis.set_ticks_position('left')
        plot.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plot.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        plot.set_yticks(ytick, fontsize=fontsize)
        plot.set_xlim([mini, maxi])
        plot.set_xticks([mini, self.center, maxi])
        plot.set_xticklabels([mini, self.center, maxi])
        plot.set_yticklabels(self.df.study, fontsize=fontsize)
        plot.yaxis.set_ticks_position('none')
        plot.invert_yaxis()  # invert y-axis to align values properly with table
        tb = tabl.table(cellText=tval, cellLoc='center', loc='right', colLabels=[self.em, self.ci, self.p], bbox=[0, t_adjuster, 3.5, 1])
        # tb = tabl.table(cellText=tval, cellLoc='center', loc='right', colLabels=[self.em, self.ci, self.p, self.subtype, self.purity, self.counts, self.mean_tp, self.max_tp, self.perc_pat], bbox=[0, t_adjuster, 4.5, 1])
        tabl.axis('off')
        tb.auto_set_font_size(False)
        tb.set_fontsize(fontsize)
        for (row, col), cell in tb.get_celld().items():
            c_pvalue = self.df['P'].values[row-1]
            if c_pvalue < p_th and row !=0:
                cell.set_text_props(fontproperties=FontProperties(size=fontsize))
            else:
                cell.set_text_props(fontproperties=FontProperties(weight='light', size=fontsize))
            if (row == 0):
                cell.set_text_props(fontproperties=FontProperties(weight='bold', size=fontsize))
            cell.set_linewidth(0)
        return plot
    
def check_class_balance(meta_field, data_res_folds, n_folds):
    print(f'{meta_field} Positive\n')
    train_class = list()
    test_class = list()
    for i in range(n_folds):
        print(f'Fold {i}')
        train_shape = data_res_folds[i]['train_y'].shape
        train_positive = np.sum(data_res_folds[i]['train_y'])
        prop_positive = np.round((train_positive / train_shape)[0], 2)
        print(f'\t Train: {prop_positive:.2f}')
        train_class.append(prop_positive)
        test_shape = data_res_folds[i]['test_y'].shape
        test_positive = np.sum(data_res_folds[i]['test_y'])
        prop_positive = np.round((test_positive / test_shape)[0], 2)
        print(f'\t Test: {prop_positive:.2f}')
        test_class.append(prop_positive)

    return np.mean(train_class), np.mean(test_class)

# Wrap up into a single function call

def run_logit_workflow(adatas_path, resolution, n_folds, meta_frame, additional_meta_frame, meta_field, add_meta_field, matching_field, oversample, h5_complete_path, h5_additional_path, scale_to_plot='exp', force_fold=2, leiden_clusters=None, min_perc=0):
    print(f'Loading data: Leiden {resolution}\n')
    groupby = f'leiden_{resolution}'
    data_res_folds = dict()

    if add_meta_field is not None:
        meta_field = add_meta_field

    if h5_additional_path is not None:
        complete_df, additional_df, _ = read_csvs_forcefold(adatas_path=adatas_path, groupby=groupby, h5_complete_path=h5_complete_path, h5_additional_path=h5_additional_path, force_fold=force_fold)
        additional_meta_df = additional_df.merge(additional_meta_frame[['samples', meta_field]], on='samples')
        additional_meta_df = additional_meta_df[~additional_meta_df[meta_field].isna()]
    else:
        additional_meta_df = None
        complete_df, _ = read_csvs_forcefold(adatas_path=adatas_path, groupby=groupby, h5_complete_path=h5_complete_path, h5_additional_path=h5_additional_path, force_fold=force_fold)
    
    # keep_clusters = [c for c in leiden_clusters if c not in remove_clusters]
    
    complete_meta_df = complete_df.merge(meta_frame[['samples', meta_field]], on='samples')
    complete_meta_df = complete_meta_df[~complete_meta_df[meta_field].isna()]
        
    data, labels, sample_rep_df = generate_frequency_vector(complete_df=complete_meta_df, matching_field=matching_field, groupby=groupby, leiden_clusters=leiden_clusters, meta_field=meta_field, min_perc=min_perc)
    if additional_meta_df is not None:
        additional_data, additional_labels, additional_sample_rep_df = generate_frequency_vector(complete_df=additional_meta_df, matching_field=matching_field, groupby=groupby, leiden_clusters=leiden_clusters, meta_field=meta_field, min_perc=min_perc)
    else:
        additional_data = None
        additional_labels = None

    skf = StratifiedKFold(n_splits=n_folds)

    for i, (train_idx, test_idx) in enumerate(skf.split(data, labels)):
        data_res_folds[i] = dict()
        data_res_folds[i]['train_X'] = data[train_idx]
        data_res_folds[i]['train_y'] = labels[train_idx]
        data_res_folds[i]['test_X'] = data[test_idx]
        data_res_folds[i]['test_y'] = labels[test_idx] 
        data_res_folds[i]['additional_X'] = additional_data
        data_res_folds[i]['additional_y'] = additional_labels

    if oversample:
        try:
            smt = SMOTETomek(random_state=42)
            print(f'\nUsing SMOTETomek to resample {meta_field}')
            for i in range(n_folds):
                data_res_folds[i]['train_X'], data_res_folds[i]['train_y'] = smt.fit_resample(data_res_folds[i]['train_X'], data_res_folds[i]['train_y'])
                # data_res_folds[i]['test_X'], data_res_folds[i]['test_y'] = smt.fit_resample(data_res_folds[i]['test_X'], data_res_folds[i]['test_y'])
        except ValueError as e:
            print(f'\nInsufficient samples for SMOTETomek -- using Random Over Sampler for {meta_field}')
            ros = RandomOverSampler(random_state=42)
            for i in range(n_folds):
                data_res_folds[i]['train_X'], data_res_folds[i]['train_y'] = ros.fit_resample(data_res_folds[i]['train_X'], data_res_folds[i]['train_y'])
                # data_res_folds[i]['test_X'], data_res_folds[i]['test_y'] = ros.fit_resample(data_res_folds[i]['test_X'], data_res_folds[i]['test_y'])

    results_frames = list()
    scores_frames = list()

    alphas = [0.1, 0.5, 1, 2.5, 5, 10, 25]
    current = generate_timestamp()
    results_dir = os.path.join("/nfs/home/users/krakovic/sharedscratch/notebooks/latticea_he/final_figures/Logit/results", f'Logit_{meta_field}_Oversample_{oversample}_{current}')
    # results_dir = os.path.join(adatas_path, f'Logit_{meta_field}_Oversample_{oversample}_{current}')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    with open(os.path.join(results_dir, 'data_res_folds.pkl'), 'wb') as outFile:
        pickle.dump(data_res_folds, outFile)

    if scale_to_plot == 'exp':
        sortby = 'exp_coef'
    else:
        sortby = 'coef'

    for alpha in alphas:
        print(f'Alpha: {alpha}')
        for i in range(n_folds):
            results_df, scores_df = run_logit(data_dict=data_res_folds, fold=i, leiden_clusters=leiden_clusters, alpha=alpha)
            results_df = results_df.sort_values(by=sortby, ascending=True)
            csv_name = os.path.join(results_dir, f'logit_{meta_field}_alpha_{alpha}_fold_{i}.csv')
            results_df.to_csv(csv_name)
            forest_plots_logit(results_df=results_df, scale_to_plot=scale_to_plot, results_dir=results_dir, csv_name=csv_name)
            # scores_df.to_csv(csv_name.replace(".csv", "_statistics.csv"))
            # results_frames.append(results_df)
            scores_frames.append(scores_df)
    
    all_scores_frames = pd.concat(scores_frames)
    all_scores_frames.to_csv(os.path.join(results_dir, f'{meta_field}_statistics_all_folds.csv'))
