"""
Compute and save tables with Cohen's kappa and Scott's pi concordance
"""

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from Coefficients import kappa_pi

indices = ['MSE', 'PSNR', 'SNR', 'WSNR', 'NQM', 'UQI', 'SSIM', 'MSSIM', 'VIF', 'CQ(1,1)', 'GMSM', 'GMSD']

groups = ['Noise', 'Actual', 'Simple', 'Exotic', 'New', 'Color', 'Full']

Cohen = {}

Scott = {}

for g in groups:

    if g != 'Full':
        df = pd.read_csv('../../Data/by group/' + g + '_categ.csv', delimiter=',')
    else:
        df = pd.read_csv('../../Data/data_categ.csv', delimiter=',')

    mos = np.asarray(df['MOS'])

    cohen_list = []

    scott_list = []

    for ind in indices:

        pred = np.asarray(df[ind])

        cohen_scott = kappa_pi(mos, pred)

        c = cohen_kappa_score(mos, pred)
        # c = cohen_scott[0] # this is equivalent

        s = cohen_scott[1]

        cohen_list.append(c)
        scott_list.append(s)

    Cohen.update({g: cohen_list})
    Scott.update({g: scott_list})

cohen = pd.DataFrame(Cohen, index=indices, columns=groups)
scott = pd.DataFrame(Scott, index=indices, columns=groups)

cohen.to_csv('../../Results/Cohen-concordance.csv', float_format='%.10f')
scott.to_csv('../../Results/Scott-concordance.csv', float_format='%.10f')
