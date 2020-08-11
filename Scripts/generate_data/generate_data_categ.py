"""
This script convert the data to categorical data
This script must be run after generate_data.py
"""

import pandas as pd
import numpy as np


def categorize(values, a, b, decreasing=False):
    """
    Turns a vector of continuous values into a vector of categorical values, according to three possible categories.

    Categories:
    1: bad quality
    2: middle quality
    3: good quality

    :param values: vector with the continuous values
    :param a: category bound (usually one-third percentile)
    :param b: category bound (usually two-third percentile)
    :param decreasing: True if the correlation with the MOS should be inversely proportional, False otherwise.
    :return: a list with 1's, 2's and 3's according to the assigned category
    """

    cat = []

    for elem in values:

        if elem < a:

            if decreasing:
                cat.append(3)
            else:
                cat.append(1)

        elif a <= elem < b:

            cat.append(2)

        elif elem >= b:

            if decreasing:
                cat.append(1)
            else:
                cat.append(3)

    return cat


def category_bounds():
    """
    :return: A list with category bounds for each quality index
    """

    df = pd.read_csv('Data/data.csv', delimiter=',')

    indices = ['MOS', 'MSE', 'RMSE', 'PSNR', 'SNR', 'WSNR', 'NQM', 'UQI', 'SSIM', 'MSSIM', 'VIF', 'CQ(1,1)', 'GMSM', 'GMSD']

    bounds = []

    for ind in indices:

        a = np.percentile(df[ind], 100 / 3)  # one-third percentile
        b = np.percentile(df[ind], 200 / 3)  # two-third percentile

        bounds.append((a, b))

    return bounds


def categorize_set(df, path):
    """
    Transform the continuous dataset into a categorical dataset
    :param df: pandas dataframe with the dataset
    :param path: path to save the new dataset
    :return: None
    """

    mos = df['MOS']
    mse = df['MSE']
    rmse = df['RMSE']
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

    dict = {}

    dict['name'] = df['name']
    dict['image'] = df['image']
    dict['distortion'] = df['distortion']
    dict['level'] = df['level']

    bounds = category_bounds()

    a, b = bounds[0]
    dict['MOS'] = categorize(mos, a, b)
    a, b = bounds[1]
    dict['MSE'] = categorize(mse, a, b, decreasing=True)
    a, b = bounds[2]
    dict['RMSE'] = categorize(rmse, a, b, decreasing=True)
    a, b = bounds[3]
    dict['PSNR'] = categorize(psnr, a, b)
    a, b = bounds[4]
    dict['SNR'] = categorize(snr, a, b)
    a, b = bounds[5]
    dict['WSNR'] = categorize(wsnr, a, b)
    a, b = bounds[6]
    dict['NQM'] = categorize(nqm, a, b)
    a, b = bounds[7]
    dict['UQI'] = categorize(uqi, a, b)
    a, b = bounds[8]
    dict['SSIM'] = categorize(ssim, a, b)
    a, b = bounds[9]
    dict['MSSIM'] = categorize(mssim, a, b)
    a, b = bounds[10]
    dict['VIF'] = categorize(vif, a, b)
    a, b = bounds[11]
    dict['CQ(1,1)'] = categorize(cq, a, b)
    a, b = bounds[12]
    dict['GMSM'] = categorize(gmsm, a, b)
    a, b = bounds[13]
    dict['GMSD'] = categorize(gmsd, a, b, decreasing=True)

    headers = ('name', 'image', 'distortion', 'level', 'MOS', 'MSE', 'RMSE', 'PSNR', 'SNR',
               'WSNR', 'NQM', 'UQI', 'SSIM', 'MSSIM', 'VIF', 'CQ(1,1)', 'GMSM', 'GMSD')

    result = pd.DataFrame(data=dict, columns=headers)

    result.to_csv(path, index=False)


df = pd.read_csv('../../Data/data.csv', delimiter=',')
path = '../../Data/data_categ.csv'

categorize_set(df, path)
