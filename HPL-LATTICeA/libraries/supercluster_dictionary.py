import numpy as np
import pandas as pd

# def count_fibroblast(row):
#     return row['cluster_5'] + row['cluster_30'] + row['cluster_64']

# def count_ecm(row):
#     return row['cluster_21'] + row['cluster_27'] + row['cluster_40']
#     # return row['cluster_5'] + row['cluster_21'] + row['cluster_27'] + row['cluster_40']

# def count_lymphocyte(row):
#     return row['cluster_10'] + row['cluster_16'] + row['cluster_28'] + row['cluster_48'] + row['cluster_61']
    
# def count_lepidic(row):
#     # return row['cluster_0'] + row['cluster_35']
#     return row['cluster_0']

# def count_acinar(row):
#     return row['cluster_11'] + row['cluster_31'] + row['cluster_44'] + row['cluster_67'] + row['cluster_68'] + row['cluster_69']

# def count_papillary(row):
#     return row['cluster_41'] + row['cluster_46'] + row['cluster_54']

# def count_cribriform(row):
#     return row['cluster_34']

# def count_solid(row):
#     return row['cluster_13'] + row['cluster_17'] + row['cluster_47'] + row['cluster_50']

# def count_mucinous(row):
#     return row['cluster_52']

# def assign_supercluster(x):
#     if x in ['HPC 5', 'HPC 30', 'HPC 64']:
#         return 'Fibroblast-rich'
#     elif x in ['HPC 21', 'HPC 27', 'HPC 40']:
#         return 'ECM-rich'
#     elif x in ['HPC 10', 'HPC 16', 'HPC 28', 'HPC 48', 'HPC 61']:
#         return 'Lymphocyte-rich'
#     elif x in ['HPC 0', 'HPC 35', 'HPC 52']:
#         return 'Lepidic'
#     elif x in ['HPC 11', 'HPC 31', 'HPC 44', 'HPC 67', 'HPC 68', 'HPC 69']:
#         return 'Acinar'
#     elif x in ['HPC 41', 'HPC 46', 'HPC 54']:
#         return 'Pap./Mpap.'
#     elif x in ['HPC 34']:
#         return 'Cribriform'
#     else:
#         return 'Solid'

def assign_supercluster(x):
    if x in ['HPC 69', 'HPC 67', 'HPC 0', 'HPC 50', 'HPC 44']:
        return 'Hot, cohesive'
    elif x in ['HPC 10', 'HPC 16', 'HPC 28', 'HPC 61', 'HPC 48', 'HPC 13', 'HPC 5', 'HPC 35', 'HPC 31']:
        return 'Hot, discohesive'
    elif x in ['HPC 33', 'HPC 21', 'HPC 47', 'HPC 17', 'HPC 11', 'HPC 68', 'HPC 34']:
        return 'Cold, cohesive'
    elif x in ['HPC 30', 'HPC 40', 'HPC 27', 'HPC 64']:
        return 'Cold, discohesive'
    else:
        return 'Unknown'
    
def assign_supercluster_v2(x):
    if x in ['HPC 67', 'HPC 0', 'HPC 50', 'HPC 44']:
        return 'Hot, cohesive'
    elif x in ['HPC 10', 'HPC 16', 'HPC 28', 'HPC 61', 'HPC 48', 'HPC 13', 'HPC 69']:
        return 'Hot, discohesive'
    elif x in ['HPC 33', 'HPC 21', 'HPC 47', 'HPC 17', 'HPC 11', 'HPC 68', 'HPC 34']:
        return 'Cold, cohesive'
    elif x in ['HPC 30', 'HPC 40', 'HPC 27', 'HPC 64', 'HPC 5', 'HPC 31']:
        return 'Cold, discohesive'
    else:
        return 'Unknown'

    
def count_hot_cohesive(row):
    try:
        c69 = row['cluster_69']
    except KeyError:
        c69 = 0
    
    try:
        c67 = row['cluster_67']
    except KeyError:
        c67 = 0

    try:
        c0 = row['cluster_0']
    except KeyError:
        c0 = 0

    try:
        c50 = row['cluster_50']
    except KeyError:
        c50 = 0

    try:
        c44 = row['cluster_44']
    except KeyError:
        c44 = 0

    return np.sum([c69, c67, c0, c50, c44])
    # return row['cluster_69'] + row['cluster_67'] + row['cluster_0'] + row['cluster_50'] + row['cluster_44']

def count_hot_discohesive(row):
    try:
        c10 = row['cluster_10']
    except KeyError:
        c10 = 0
    
    try:
        c16 = row['cluster_16']
    except KeyError:
        c16 = 0

    try:
        c28 = row['cluster_28']
    except KeyError:
        c28 = 0

    try:
        c61 = row['cluster_61']
    except KeyError:
        c61 = 0

    try:
        c48 = row['cluster_48']
    except KeyError:
        c48 = 0

    try:
        c13 = row['cluster_13']
    except KeyError:
        c13 = 0

    try:
        c5 = row['cluster_5']
    except KeyError:
        c5 = 0

    try:
        c35 = row['cluster_35']
    except KeyError:
        c35 = 0

    try:
        c31 = row['cluster_31']
    except KeyError:
        c31 = 0

    return np.sum([c10, c16, c28, c61, c48, c13, c5, c35, c31])
    # return row['cluster_10'] + row['cluster_16'] + row['cluster_28'] + row['cluster_61'] + row['cluster_48'] + row['cluster_13'] + row['cluster_5'] + row['cluster_35'] + row['cluster_31']

def count_cold_cohesive(row):
    try:
        c33 = row['cluster_33']
    except KeyError:
        c33 = 0

    try:
        c21 = row['cluster_21']
    except KeyError:
        c21 = 0

    try:
        c47 = row['cluster_47']
    except KeyError:
        c47 = 0

    try:
        c17 = row['cluster_17']
    except KeyError:
        c17 = 0

    try:
        c11 = row['cluster_11']
    except KeyError:
        c11 = 0

    try:
        c68 = row['cluster_68']
    except KeyError:
        c68 = 0

    try:
        c34 = row['cluster_34']
    except KeyError:
        c34 = 0

    return np.sum([c33, c21, c47, c17, c11, c68, c34])
    # return row['cluster_33'] + row['cluster_21'] + row['cluster_47'] + row['cluster_17'] + row['cluster_11'] + row['cluster_68'] + row['cluster_34']

def count_cold_discohesive(row):
    try:
        c30 = row['cluster_30']
    except KeyError:
        c30 = 0

    try:
        c40 = row['cluster_40']
    except KeyError:
        c40 = 0

    try:
        c27 = row['cluster_27']
    except KeyError:
        c27 = 0

    try:
        c64 = row['cluster_64']
    except KeyError:
        c64 = 0

    return np.sum([c30, c40, c27, c64])
    # return row['cluster_30'] + row['cluster_40'] + row['cluster_27'] + row['cluster_64']