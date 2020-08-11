"""
This script convert the data to categorical data for each distortion group
This script must be run after generate_group_data.py
"""

import pandas as pd
from generate_data_categ import categorize_set

groups = ['Noise', 'Actual', 'Simple', 'Exotic', 'New', 'Color']

for g in groups:

    df = pd.read_csv('../../Data/by group/{group}.csv'.format(group=g), delimiter=',')
    path = '../../Data/by group/' + g + '_categ.csv'

    categorize_set(df, path)
