"""
Compute and save tables with Spearman and Kendall correlations and p-values of respective tests
"""

import pandas as pd
from scipy.stats import spearmanr, kendalltau

indices = ['MSE', 'PSNR', 'SNR', 'WSNR', 'NQM', 'UQI', 'SSIM', 'MSSIM', 'VIF', 'CQ(1,1)', 'GMSM', 'GMSD']

groups = ['Noise', 'Actual', 'Simple', 'Exotic', 'New', 'Color', 'Full']

S = {}
pS = {}

K = {}
pK = {}

for g in groups:

    if g != 'Full':
        df = pd.read_csv('../../Data/by group/' + g + '.csv', delimiter=',')
    else:
        df = pd.read_csv('../../Data/data.csv', delimiter=',')

    mos = df['MOS']

    gScorrs = []
    gSpval = []

    gKcorrs = []
    gKpval = []

    for ind in indices:

        s = spearmanr(mos, df[ind])

        gScorrs.append(s[0])
        gSpval.append(s[1])

        k = kendalltau(mos, df[ind])

        gKcorrs.append(k[0])
        gKpval.append(k[1])

    S.update({g: gScorrs})
    pS.update({g: gSpval})

    K.update({g: gKcorrs})
    pK.update({g: gKpval})

Scorr = pd.DataFrame(S, index=indices, columns=groups)
Spval = pd.DataFrame(pS, index=indices, columns=groups)

Kcorr = pd.DataFrame(K, index=indices, columns=groups)
Kpval = pd.DataFrame(pK, index=indices, columns=groups)

Scorr.to_csv('../../Results/Spearman-correlation.csv', float_format='%.10f')
Spval.to_csv('../../Results/Spearman-pvalues.csv', float_format='%.20f')

Kcorr.to_csv('../../Results/Kendall-correlation.csv', float_format='%.10f')
Kpval.to_csv('../../Results/Kendall-pvalues.csv', float_format='%.20f')
