"""
Compute and save Fleiss' kappa multivariate concordance coefficient
"""

import pandas as pd
from Coefficients import kappa_fleiss

groups = ['Noise', 'Actual', 'Simple', 'Exotic', 'New', 'Color', 'Full']

K = {}

k_list = []

k2_list = []

for g in groups:

    if g != 'Full':
        df = pd.read_csv('../../Data/by group/' + g + '_categ.csv', delimiter=',')
    else:
        df = pd.read_csv('../../Data/data_categ.csv', delimiter=',')

    mos = df['MOS']
    mse = df['MSE']
    psnr = df['PSNR']
    snr = df['SNR']
    wsnr = df['WSNR']
    nqm = df['NQM']
    uqi = df['UQI']
    ssim = df['SSIM']
    mssim = df['MSSIM']
    vif = df['VIF']
    cq = df['CQ(1,1)']
    gmsm = df['GMSM']
    gmsd = df['GMSD']

    k = kappa_fleiss(mos, mse, psnr, snr, wsnr, nqm, uqi, ssim, mssim, vif, cq, gmsm, gmsd)

    k_list.append(k)

    if g == 'Noise':
        k2 = kappa_fleiss(mos, gmsd, gmsm, wsnr)

    elif g == 'Actual' or g == 'Exotic' or g == 'Full':
        k2 = kappa_fleiss(mos, gmsd, gmsm, mssim)

    elif g == 'Simple':
        k2 = kappa_fleiss(mos, gmsd, gmsm, mssim, wsnr)

    elif g == 'New' or g == 'Color':
        k2 = kappa_fleiss(mos, nqm, wsnr, gmsd, mssim)

    k2_list.append(k2)

K.update({'k_all_indices_MOS': k_list})
K.update({'k_subset_indices_MOS': k2_list})

wdf = pd.DataFrame(K, index=groups, columns=['k_all_indices_MOS', 'k_subset_indices_MOS'])

wdf.to_csv('../../Results/Fleiss-K.csv', float_format='%.10f')
