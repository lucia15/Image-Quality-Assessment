"""
This script generate the data for each distortion group
This script must be run after generate_data.py
"""

import pandas as pd


groups = {'Noise': [1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 21],
          'Actual': [1, 3, 4, 5, 6, 8, 9, 19, 21],
          'Simple': [1, 8, 10],
          'Exotic': [12, 13, 14, 15, 16, 17, 20, 23, 24],
          'New': [18, 19, 20, 21, 22, 23, 24],
          'Color': [2, 7, 10, 18, 23, 24]}

headers = ('name', 'image', 'distortion', 'level', 'MOS', 'MOS_std', 'MSE', 'RMSE', 'PSNR',
           'SNR', 'WSNR', 'NQM', 'UQI', 'SSIM', 'MSSIM', 'VIF', 'CQ(1,1)', 'GMSM', 'GMSD')

for key in groups:

    frames = []

    for n in groups[key]:

        df = pd.read_csv('../../Data/by distortion/distortion{i}.csv'.format(i=n), delimiter=',')

        frames.append(df)

    result = pd.concat(frames)

    result.to_csv('../../Data/by group/' + key + '.csv', index=False, float_format='%.5f')
