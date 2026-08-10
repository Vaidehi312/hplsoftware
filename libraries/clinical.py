import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import normalize


def load_clinical(main_path=os.path.abspath(os.path.join(os.getcwd(), '..'))):
    clinpath = pd.read_csv(os.path.join(main_path, 'utilities/files/LUAD-LATTICeA/latticea_master_clinicopathological.csv'))

    clinpath = clinpath.sort_values(by='Case Number').reset_index()
    clinpath['Case Number'] = clinpath['Case Number'].astype(str)

    samples_list = []

    for i in range(1, 5):
        indexes = clinpath[clinpath['Case Number'].str.len() == i].index
        if i == 1:
            samples_list.extend(['ACA_000' + str(x) for x in clinpath.iloc[indexes]['Case Number']])
        elif i == 2:
            samples_list.extend(['ACA_00' + str(x) for x in clinpath.iloc[indexes]['Case Number']])
        elif i == 3:
            samples_list.extend(['ACA_0' + str(x) for x in clinpath.iloc[indexes]['Case Number']])
        else:
            samples_list.extend(['ACA_' + str(x) for x in clinpath.iloc[indexes]['Case Number']])

    clinpath['samples'] = samples_list

    ## Isolate the growth pattern proportions according to pathologist 1 assessment

    patterns = clinpath[['In Situ Proportion (%)', 'Acinar Proportion (%)', 'Papillary Proportion (%)', 'Cribriform Proportion (%)', 'Solid Proportion (%)', 'Micropapillary Proportion (%)']]
    patterns = patterns.replace(np.nan, 0)
    X_patterns = patterns.values
    X_patterns = normalize(X_patterns, axis=1, norm='l1')
    patterns = pd.DataFrame(X_patterns, columns=patterns.columns)

    for i in patterns.index:
        solid_proportion = patterns.loc[i, 'Solid Proportion (%)']
        cribriform_proportion = patterns.loc[i, 'Cribriform Proportion (%)']
        micropapillary_proportion = patterns.loc[i, 'Micropapillary Proportion (%)']
        acinar_proportion = patterns.loc[i, 'Acinar Proportion (%)']
        papillary_proportion = patterns.loc[i, 'Papillary Proportion (%)']
        lepidic_proportion = patterns.loc[i, 'In Situ Proportion (%)']

        if solid_proportion + cribriform_proportion + micropapillary_proportion >= 0.2:
            patterns.at[i, 'iaslc_grade_'] = 'G3'
        elif acinar_proportion + papillary_proportion > lepidic_proportion:
            patterns.at[i, 'iaslc_grade_'] = 'G2'
        else:
            patterns.at[i, 'iaslc_grade_'] = 'G1'

    patterns_samples = patterns[['iaslc_grade_']].merge(clinpath[['samples']], left_index=True, right_index=True)
    patterns_samples['iaslc_grade'] = patterns_samples['iaslc_grade_'].map({'G1':1, 'G2':2, 'G3':3})
    patterns_samples['iaslc_grade'] = pd.Categorical(patterns_samples['iaslc_grade'], categories=[1, 2, 3], ordered=True) 

    survival = clinpath.rename(columns={'Time to Recurrence-Free Survival Status (Days)': 'rfs_event_data',
                                        'Recurrence-Free Survival Status':'rfs_event_ind',
                                        'Time to Survival Status (Days)':'os_event_data',
                                        'Survival Status':'os_event_ind',
                                        'Age at Surgery':'age',
                                        'Sex':'male',
                                        'Overall Stage (8th TNM Edition)':'stage',
                                        'Pleural Involvement':'pl_stage',
                                        '2015 WHO Classification':'2015_who',
                                        'Vascular Invasion.1':'lvi',
                                        'In Situ Proportion (%)':'lepidic', 
                                        'Acinar Proportion (%)':'acinar', 
                                        'Papillary Proportion (%)':'papillary', 
                                        'Cribriform Proportion (%)':'cribriform', 
                                        'Solid Proportion (%)':'solid', 
                                        'Micropapillary Proportion (%)':'micropapillary',
                                        'Pack Years':'pack_years',
                                        'PD-L1 Stained Percentage':'PDL1_score'})

    survival = survival.replace({'IA1': 'I', 'IA2':'I', 'IA3':'I', 'IB':'I', 
                                'IIA': 'II', 'IIB':'II',
                                'IIIA':'III', 'IIIB':'III', 'IIIC':'III', 
                                'IVA':'IV'})

    survival = survival.replace({'MALE': 1, 'FEMALE': 0})

    survival = survival.replace({'No recurrence':0, 'Recurrence':1})

    survival = survival.replace({'Dead':1, 'Alive': 0})

    survival = survival.dropna(subset='stage')
    survival = survival[survival['stage'] != '0']
    survival.replace({'III':'III-IV',
                    'IV':'III-IV'}, inplace=True)
    survival['stage_num'] = survival['stage'].map({'I':1, 'II':2, 'III-IV':3})
    survival['stage'] = pd.Categorical(survival['stage'], categories=['I', 'II', 'III-IV'], ordered=True)
    survival['stage_num'] = pd.Categorical(survival['stage_num'], categories=[1, 2, 3], ordered=True)

    pdl1_positive_dict = {'0%':0, '1-49%':1, '50-100%':1}

    survival['PDL_positive'] = survival['PDL1_score'].map(pdl1_positive_dict)
    survival = survival[survival['2015_who'] != 'Invasive mucinous adenocarcinoma (IMA)']

    return clinpath, survival, patterns_samples

def load_genetics(main_path=os.path.abspath(os.path.join(os.getcwd(), '..'))):
    genetics = pd.read_csv(os.path.join(main_path, 'utilities/files/LUAD-LATTICeA/LatticeDatabase_withgenetics.tsv'), sep='\t')
    genetics['samples'] = genetics['case_num'].apply(lambda x: f"ACA_{x:04d}")

    return genetics