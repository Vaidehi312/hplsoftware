from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test
from lifelines.statistics import pairwise_logrank_test
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns

def train_cox(data_dict, fold,  penalizer, l1_ratio, event_data_col, event_ind_col, cols_to_include, additional_flag=True, robust=True, save_path=None):
    train, test = data_dict[fold]['train_all'], data_dict[fold]['test_all']
    if additional_flag:
        additional = data_dict[fold]['additional_all']

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