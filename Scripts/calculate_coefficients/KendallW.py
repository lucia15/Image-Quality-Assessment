"""
Compute and save Kendall's w concordance coefficient and p-value of test
"""

import pandas as pd
from Coefficients import kendall_w

groups = ['Noise', 'Actual', 'Simple', 'Exotic', 'New', 'Color', 'Full']

W = {}
pW = {}

w_list = []
p_list = []

w2_list = []
p2_list = []

for g in groups:

    if g != 'Full':
        df = pd.read_csv('../../Data/by group/' + g + '.csv', delimiter=',')
    else:
        df = pd.read_csv('../../Data/data.csv', delimiter=',')

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

    w = kendall_w(mos, -1 * mse, psnr, snr, wsnr, nqm, uqi, ssim, mssim, vif, cq, gmsm, -1 * gmsd)

    w_list.append(w[0][0])
    p_list.append(w[1][0])

    if g == 'Noise':
        w2 = kendall_w(mos, -1 * gmsd, gmsm, wsnr)

    elif g == 'Actual' or g == 'Exotic' or g == 'Full':
        w2 = kendall_w(mos, -1 * gmsd, gmsm, mssim)

    elif g == 'Simple':
        w2 = kendall_w(mos, -1 * gmsd, gmsm, mssim, wsnr)

    elif g == 'New' or g == 'Color':
        w2 = kendall_w(mos, nqm, wsnr, -1 * gmsd, mssim)

    w2_list.append(w2[0][0])
    p2_list.append(w2[1][0])

W.update({'w_all_indices_MOS': w_list})
W.update({'w_subset_indices_MOS': w2_list})

pW.update({'p-value_all_indices_MOS': p_list})
pW.update({'p-value_subset_indices_MOS': p2_list})

wdf = pd.DataFrame(W, index=groups, columns=['w_all_indices_MOS', 'w_subset_indices_MOS'])
pdf = pd.DataFrame(pW, index=groups, columns=['p-value_all_indices_MOS', 'p-value_subset_indices_MOS'])

wdf.to_csv('../../Results/Kendall-W.csv', float_format='%.10f')
pdf.to_csv('../../Results/Kendall-Wtest-pvalues.csv', float_format='%.20f')
