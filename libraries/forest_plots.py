import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import links
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.font_manager import FontProperties
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import os

import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import links
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.font_manager import FontProperties
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import os

class EffectMeasurePlot_Cox:
    """Used to generate effect measure plots. effectmeasure plot accepts four list type objects.
    effectmeasure_plot is initialized with the associated names for each line, the point estimate,
    the lower confidence limit, and the upper confidence limit.
    Plots will resemble the following form:
        _____________________________________________      Measure     % CI
        |                                           |
    1   |        --------o-------                   |       x        n, 2n
        |                                           |
    2   |                   ----o----               |       w        m, 2m
        |                                           |
        |___________________________________________|
        #           #           #           #
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
    Example)
    >>>lab = ['One','Two'] #generating lists of data to plot
    >>>emm = [1.01,1.31]
    >>>lcl = ['0.90',1.01]
    >>>ucl = [1.11,1.53]
    >>>
    >>>x = zepid.graphics.effectmeasure_plot(lab,emm,lcl,ucl) #initializing effectmeasure_plot with the above lists
    >>>x.labels(effectmeasure='RR') #changing the table label to 'RR'
    >>>x.colors(pointcolor='r') #changing the point colors to red
    >>>x.plot(t_adjuster=0.13) #generating the effect measure plot
    """

    def __init__(self, label, effect_measure, lcl, ucl, pvalues, center=0):
        self.df = pd.DataFrame()
        self.df['study'] = label
        self.df['OR']    = effect_measure
        self.df['LCL']   = lcl
        self.df['UCL']   = ucl
        self.df['P']     = pvalues
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
        # self.counts   = 'Tile Counts'
        # self.mean_tp  = 'Mean Tile Pat.'
        # self.max_tp   = 'Max Tile Pat. %'
        # self.perc_pat = 'Patients %'
        self.scale    = 'linear'
        self.center   = center
        self.errc     = 'dimgrey'
        self.shape    = 'o'
        self.pc       = 'k'
        self.linec    = 'gray'

    def labels(self, **kwargs):
        if 'effectmeasure' in kwargs:
            self.em = kwargs['effectmeasure']
        if 'ci' in kwargs:
            self.ci = kwargs['conf_int']
        if 'scale' in kwargs:
            self.scale = kwargs['scale']
        if 'center' in kwargs:
            self.center = kwargs['center']

    def colors(self, **kwargs):
        if 'errorbarcolor' in kwargs:
            self.errc = kwargs['errorbarcolor']
        if 'pointshape' in kwargs:
            self.shape = kwargs['pointshape']
        if 'linecolor' in kwargs:
            self.linec = kwargs['linecolor']
        if 'pointcolor' in kwargs:
            self.pc = kwargs['pointcolor']

    def plot(self, figsize=(3, 3), t_adjuster=0.01, decimal=3, size=50, max_value=None, min_value=None, fontsize=None, p_th=0.05):
        tval = []
        ytick = []
        for i in range(len(self.df)):
            if (np.isnan(self.df['OR2'][i]) == False):
                if ((isinstance(self.df['OR'][i], float)) & (isinstance(self.df['LCL'][i], float)) & (isinstance(self.df['UCL'][i], float))):
                    # list_val = [round(self.df['OR2'][i], decimal), ('(' + str(round(self.df['LCL'][i], decimal)) + ', ' + str(round(self.df['UCL'][i], decimal)) + ')'), str(self.df['P'][i]), \
                    #             self.df['C'][i], self.df['M'][i], self.df['Ma'][i], self.df['Pp'][i]]
                    list_val = [round(self.df['OR2'][i], decimal), ('(' + str(round(self.df['LCL'][i], decimal)) + ', ' + str(round(self.df['UCL'][i], decimal)) + ')'), str(self.df['P'][i])]
                    tval.append(list_val)
                else:
                    list_val = [self.df['OR'][i], ('(' + str(self.df['LCL'][i]) + ', ' + str(self.df['UCL'][i]) + ')'), self.df['P'][i], self.df['C'][i]]
                    tval.append()
                ytick.append(i)
            else:
                tval.append([' ', ' ', ' ', ' '])
                ytick.append(i)
        if max_value is None:
            if pd.to_numeric(self.df['UCL']).max() < 1:
                maxi = round(((pd.to_numeric(self.df['UCL'])).max() + 0.05),
                             3)  # setting x-axis maximum for UCL less than 1
            if (pd.to_numeric(self.df['UCL']).max() < 9) and (pd.to_numeric(self.df['UCL']).max() >= 1):
                maxi = round(((pd.to_numeric(self.df['UCL'])).max() + 1),
                             3)  # setting x-axis maximum for UCL less than 10
            if pd.to_numeric(self.df['UCL']).max() > 9:
                maxi = round(((pd.to_numeric(self.df['UCL'])).max() + 10),
                             3)  # setting x-axis maximum for UCL less than 100
        else:
            maxi = max_value
        if min_value is None:
            if pd.to_numeric(self.df['LCL']).min() > 0:
                mini = round(((pd.to_numeric(self.df['LCL'])).min() - 0.1), 3)  # setting x-axis minimum
            if pd.to_numeric(self.df['LCL']).min() < 0:
                mini = round(((pd.to_numeric(self.df['LCL'])).min() - 0.05), 3)  # setting x-axis minimum
        else:
            mini = min_value
        # mini = None
        # maxi = None
        plt.figure(figsize=figsize)  # blank figure
        gspec = gridspec.GridSpec(1, 3)  # sets up grid
        plot = plt.subplot(gspec[0, 0:2])  # plot of data
        tabl = plt.subplot(gspec[0, 2:])  # table of OR & CI
        plot.set_ylim(-1, (len(self.df)))  # spacing out y-axis properly
        if self.scale == 'log':
            try:
                plot.set_xscale('log')
            except:
                raise ValueError('For the log scale, all values must be positive')
        plot.axvline(self.center, color=self.linec, zorder=1)
        plot.errorbar(self.df.OR2, self.df.index, xerr=[self.df.LCL_dif, self.df.UCL_dif], marker='None', zorder=2, ecolor='black', elinewidth=(size * 0.25), linewidth=0, capsize=(size*0.4))
        plot.scatter(self.df.OR2, self.df.index, c=self.pc, s=(size * 10), marker=self.shape, zorder=3, edgecolors='None')
        plot.xaxis.set_ticks_position('bottom')
        plot.yaxis.set_ticks_position('left')
        plot.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plot.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        plot.set_yticks(ytick, fontsize=fontsize)
        plot.set_xlim([mini, maxi])
        plot.set_xticks([mini, self.center, maxi], fontsize=fontsize)
        plot.set_xticklabels([mini, self.center, maxi])
        plot.set_yticklabels(self.df.study, fontsize=fontsize)
        plot.yaxis.set_ticks_position('none')
        plot.invert_yaxis()  # invert y-axis to align values properly with table
        tb = tabl.table(cellText=tval, cellLoc='center', loc='right', colLabels=[self.em, self.ci, self.p], bbox=[0, t_adjuster, 3.5, 1])
        tabl.axis('off')
        tb.auto_set_font_size(False)
        tb.set_fontsize(fontsize)
        for (row, col), cell in tb.get_celld().items():
            c_pvalue = self.df['P'].values[row-1]
            # if c_pvalue < p_th and row !=0:
                # cell.set_text_props(fontproperties=FontProperties(size=fontsize))
            # else:
                # cell.set_text_props(fontproperties=FontProperties(weight='light', size=fontsize))
            if (row == 0):
                cell.set_text_props(fontproperties=FontProperties(weight='bold', size=fontsize))
            cell.set_linewidth(0)
        return plot

# Create figure with summary of all fold hazard ratios.
def summary_cox_forest_plots(estimators, cis, alpha, alphas, confidence_interval, additional_confidence_interval, groupby, alpha_path, force_fold):
    sns.set_theme(style='darkgrid')
    mosaic = '''123
                456'''
    fig = plt.figure(figsize=(20,20), constrained_layout=True)
    ax_dict = fig.subplot_mosaic(mosaic, sharex=False, sharey=False)

    for i, estimator in enumerate(estimators):
        ax_dict[str(i+1)].set_title('Fold %s - Alpha %s\nC-Index %s' % (i, np.round(alpha[i],2), np.round(cis[i][2],2)))
        estimator.plot(ax=ax_dict[str(i+1)])

    if force_fold is not None:
        ax_dict[str(6)].plot(alphas, confidence_interval[:,0], label='Test Mean')
        ax_dict[str(6)].fill_between(alphas, confidence_interval[:,1], confidence_interval[:,2], alpha=.15, label='Test CI')
        if additional_confidence_interval is not None:
            ax_dict[str(6)].plot(alphas, additional_confidence_interval[:,0], label='Additional Mean')
            ax_dict[str(6)].fill_between(alphas, additional_confidence_interval[:,1], confidence_interval[:,2], alpha=.15, label='Additional CI')
        ax_dict[str(6)].legend(loc='upper left')
        ax_dict[str(6)].set_title(groupby, fontweight='bold')
        ax_dict[str(6)].set_xscale("log")
        ax_dict[str(6)].set_ylabel("concordance index")
        ax_dict[str(6)].set_xlabel("alpha")
        ax_dict[str(6)].axvline(alpha[0], c="C1")
        ax_dict[str(6)].axhline(0.5, color="grey", linestyle="--")
        ax_dict[str(6)].grid(True)
        ax_dict[str(6)].set_ylim([0.25, 1.0])
    else:
        all_data = [(a[2], 'Test') for a in cis]
        if additional_confidence_interval is not None:
            all_data.extend([(a[3], 'Additional') for a in cis])
        all_data = pd.DataFrame(all_data, columns=['C-Index', 'Set'])
        sns.pointplot(data=all_data, y='C-Index', x='Set', ax=ax_dict[str(6)], linewidth=0.01, dodge=.3, join=False, capsize=.04, markers='s', ci=95)

    fig_path = os.path.join(alpha_path, 'hazard_ratios_summary.jpg')
    plt.savefig(fig_path)

def report_forest_plot_cph(meta_field, frame_clusters, path_csv, p_th=0.05):
    frame = frame_clusters.copy(deep=True)
    frame = frame.drop(frame[frame['coef'].isna()].index)

    sns.set_theme(style='white')
    groupby   = [value for value in frame.columns if 'leiden' in value][0]
    labs      = frame[groupby].values.tolist()
    measure   = np.round(frame['coef'],3).values.tolist()
    lower     = np.round(frame['coef lower 95%'],3).values.tolist()
    upper     = np.round(frame['coef upper 95%'],3).values.tolist()
    pvalues   = np.round(frame['p'],3).values.tolist()
    subtype   = frame['Subtype'].values.tolist()
    purity    = np.round(frame['Subtype Purity(%)'].values,1).tolist()
    counts    = frame['Subtype Counts'].values.tolist()
    mean_tp   = frame['mean_tile_sample'].values.astype(int).tolist()
    max_tp    = np.round(frame['max_tile_sample'].values*100,1).tolist()
    perc_pat  = np.round(frame['percent_sample'].values*100,1).tolist()
    max_value = max(abs(max(upper)), abs(min(lower)))

    fontsize = 14
    # try:
    #     p = EffectMeasurePlot_LR(label=labs, effect_measure=measure, lcl=lower, ucl=upper, pvalues=pvalues, subtypes=subtype, purities=purity, counts=counts, mean_tp=mean_tp, max_tp=max_tp, perc_pat=perc_pat)
    #     p.labels(effectmeasure='Log Hazard Ratio')
    #     p.colors(pointshape="o")
    #     ax=p.plot(figsize=(15,20), t_adjuster=0.01, max_value=max_value, min_value=-max_value, fontsize=fontsize, p_th=p_th)
    #     plt.suptitle("Cluster",x=0.1,y=0.89, fontsize=fontsize, fontweight='bold')
    #     ax.set_xlabel("Favours no %s              Favours %s" % (meta_field, meta_field), fontsize=fontsize, x=0.5)
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['bottom'].set_visible(True)
    #     ax.spines['left'].set_visible(False)
    #     plt.savefig(path_csv.replace('.csv', '_subytpe.jpg'),bbox_inches='tight', dpi=300)
    #     plt.close()

    p = EffectMeasurePlot_Cox(label=labs, effect_measure=measure, lcl=lower, ucl=upper, pvalues=pvalues, counts=counts, mean_tp=mean_tp, max_tp=max_tp, perc_pat=perc_pat)
    p.labels(effectmeasure='Log Hazard Ratio')
    p.colors(pointshape="o")
    ax=p.plot(figsize=(15,20), t_adjuster=0.01, max_value=max_value, min_value=-max_value, fontsize=fontsize, p_th=p_th)
    plt.suptitle("Cluster",x=0.1,y=0.89, fontsize=fontsize, fontweight='bold')
    ax.set_xlabel("Favours no %s              Favours %s" % (meta_field, meta_field), fontsize=fontsize, x=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    plt.savefig(path_csv.replace('.csv', '.jpg'),bbox_inches='tight', dpi=300)
    plt.close()
    # except:
    #     print('\t\tForest Plot - Issue with', path_csv)

def forest_plots(frame, scale_to_plot, fold, event_ind_col, save_path, figsize=(10,18), xlim=[0.7,1.3]):

    frame = frame.drop(frame[frame['coef'].isna()].index)
    frame = frame.reset_index()
    sns.set_theme(style='white')
    sns.set_context(context='paper', font_scale=2.0)
    
    labs = frame['covariate'].values.tolist()
    pvalues   = np.round(frame['p'],3).values.tolist()
    pvalues  = [p if p > 0.001 else '< 0.001' for p in pvalues]
    
    if scale_to_plot == 'exp':
        measure   = np.round(frame['exp(coef)'],3).values.tolist()
        lower     = np.round(frame['exp(coef) lower 95%'],3).values.tolist()
        upper     = np.round(frame['exp(coef) upper 95%'],3).values.tolist()
        center    = 1
        measure_lab = 'Hazard Ratio'
        max_coef  = np.round(np.max(upper), 1) 
        u_xlim    = (round(max_coef * 2) / 2) + 0.5
        # xlim      = [0.5, 1.5]
        xlim      = xlim
        # xlim      = [0, u_xlim]
        xticks    = np.round(np.arange(xlim[0], xlim[1] + 0.1, 0.1),1)
        # xticks    = np.arange(0.5, u_xlim+0.5, 0.5)

    elif scale_to_plot == 'log':
        measure   = np.round(frame['coef'],3).values.tolist()
        lower     = np.round(frame['coef lower 95%'],3).values.tolist()
        upper     = np.round(frame['coef upper 95%'],3).values.tolist()
        center    = 0
        measure_lab = 'Log Hazard Ratio'
        max_coef  = np.max(upper)
        u_xlim    = (round(max_coef * 2) / 2)
        # xlim      = [-u_xlim, u_xlim]
        xlim      = [-0.2, 0.2]
        xticks    = np.round(np.arange(xlim[0]-0.1, xlim[1]+0.1, 0.1), 2)
        # xticks    = [-1, -0.5, 0, 0.5, 1.0]
    else:
        raise Exception("Specify either 'log' or 'exp' scale")
    
    p = EffectMeasurePlot_Cox(label=labs, effect_measure=measure, lcl=lower, ucl=upper, pvalues=pvalues, center=center)
    p.labels(effectmeasure=measure_lab)
    p.colors(pointshape="o")
    
    ax = p.plot(figsize=figsize, size=12, t_adjuster=0.015, max_value=xlim[1], min_value=xlim[0], p_th=0.05)
    # plt.suptitle("Cluster",x=0.11,y=0.89, fontweight='bold')
    ax.set_xlabel(measure_lab, x=0.5)
    ax.set_xlim(xlim)
    ax.set_xscale('log')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)

    plt.gcf()
    if save_path is not None:
        fname = os.path.join(save_path, f'{event_ind_col}_fold_{fold}.csv')
        fname = fname.replace('.csv', '.jpg')
        plt.savefig(fname=fname, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()

def summary_forest_plots(frame, scale_to_plot, fold, event_ind_col, save_path):
    # frame = frame.drop(frame[frame['coef'].isna()].index)
    # frame = frame.reset_index()
    sns.set_theme(style='white')
    sns.set_context(context='paper', font_scale=2.0)
    
    labs = frame['covariate'].values.tolist()
    pvalues   = np.round(frame['mean_p'],3).values.tolist()
    
    if scale_to_plot == 'exp':
        measure   = np.round(frame['mean_exp_coef'],3).values.tolist()
        lower     = np.round(frame['exp(coef) lower 95%'],3).values.tolist()
        upper     = np.round(frame['exp(coef) upper 95%'],3).values.tolist()
        center    = 1
        measure_lab = 'Hazard Ratio'
        max_coef  = np.max(upper) 
        u_xlim    = (round(max_coef * 2) / 2)
        xlim      = [0.8, 1.3]
        xticks    = np.round(np.arange(xlim[0]-0.1, xlim[1]+0.1, 0.1),1)
        # xticks    = np.arange(0.5, u_xlim+0.5, 0.5)

    elif scale_to_plot == 'log':
        measure   = np.round(frame['coef'],3).values.tolist()
        lower     = np.round(frame['coef lower 95%'],3).values.tolist()
        upper     = np.round(frame['coef upper 95%'],3).values.tolist()
        center    = 0
        measure_lab = 'Log Hazard Ratio'
        max_coef  = np.max(upper)
        u_xlim    = (round(max_coef * 2) / 2)
        # xlim      = [-u_xlim, u_xlim]
        xlim      = [-0.2, 0.2]
        xticks    = np.round(np.arange(xlim[0]-0.1, xlim[1]+0.1, 0.1), 2)
        # xticks    = [-1, -0.5, 0, 0.5, 1.0]
    else:
        raise Exception("Specify either 'log' or 'exp' scale")
    
    p = EffectMeasurePlot_Cox(label=labs, effect_measure=measure, lcl=lower, ucl=upper, pvalues=pvalues, center=center)
    p.labels(effectmeasure=measure_lab)
    p.colors(pointshape="o")
    
    ax = p.plot(figsize=(10,18), size=8, t_adjuster=0.015, max_value=xlim[1], min_value=xlim[0], p_th=0.05)
    # plt.suptitle("Cluster",x=0.11,y=0.89, fontweight='bold')
    ax.set_xlabel(measure_lab, x=0.5)
    ax.set_xlim(xlim)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)

    plt.gcf()
    if save_path is not None:
        fname = os.path.join(save_path, f'{event_ind_col}_fold_{fold}.csv')
        fname = fname.replace('.csv', '.jpg')
        plt.savefig(fname=fname, bbox_inches='tight')
        plt.close()
    else:
        plt.show()