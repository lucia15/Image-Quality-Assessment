"""
This script apply the quality indices to each image and save the results to a csv file
"""

import numpy as np
import cv2
import csv
from os import listdir
from os.path import isfile, join
from IQA_indices import mse, rmse, psnr, snr, wsnr, nqm, uqi, ssim, mssim, pbvif, cq, gms


path = '/home/lucia/ImageDataBases/TID2013'

folders = {'reference_images', 'distorted_images'}

for folder in folders:

    path_to = join(path, folder)
    onlyfiles = [f for f in listdir(path_to) if isfile(join(path_to, f))]
    onlyfiles.sort(key=lambda v: v.lower())

    if folder == 'reference_images':
        reference_images = np.empty(len(onlyfiles), dtype=object)
        for n in range(len(onlyfiles)):
            reference_images[n] = cv2.imread(join(path_to, onlyfiles[n])) # add 0 for gray scale

    elif folder == 'distorted_images':
        distorted_images = np.empty(len(onlyfiles), dtype=object)
        for n in range(len(onlyfiles)):
            distorted_images[n] = cv2.imread(join(path_to, onlyfiles[n]))

num_ima = 25
num_dist = 24
num_lev = 5

num_dist_ima = num_dist * num_lev
num_indices = 13

table = np.empty((num_ima * num_dist_ima, 6 + num_indices))

headers = ('name', 'image', 'distortion', 'level', 'MOS', 'MOS_std', 'MSE', 'RMSE', 'PSNR',
           'SNR', 'WSNR', 'NQM', 'UQI', 'SSIM', 'MSSIM', 'VIF', 'CQ(1,1)', 'GMSM', 'GMSD')

for i in range(num_ima):

    for j in range(i * num_dist_ima, i * num_dist_ima + num_dist_ima):

        table[j, 1] = i + 1  # save image number to second column

        # save distortion number to second column
        for d in range(1, num_dist + 1):
            for k in range(i * num_dist_ima + (d - 1) * num_lev, i * num_dist_ima + d * num_lev):
                table[k, 2] = d

        # save level number to fourth column
        table[j, 3] = num_lev if (j - i * num_dist_ima + 1) % num_lev == 0 else (j - i * num_dist_ima + 1) % num_lev

        # from the seventh column onwards the quality value according to each index
        table[j, 6] = mse(reference_images[i], distorted_images[j])
        table[j, 7] = rmse(reference_images[i], distorted_images[j])
        table[j, 8] = psnr(reference_images[i], distorted_images[j])
        table[j, 9] = snr(reference_images[i], distorted_images[j])
        table[j, 10] = wsnr(reference_images[i], distorted_images[j])
        table[j, 11] = nqm(reference_images[i], distorted_images[j])
        table[j, 12] = uqi(reference_images[i], distorted_images[j])
        table[j, 13] = ssim(reference_images[i], distorted_images[j])
        table[j, 14] = mssim(reference_images[i], distorted_images[j])
        table[j, 15] = pbvif(reference_images[i], distorted_images[j])
        table[j, 16] = cq(reference_images[i], distorted_images[j])
        table[j, 17] = gms(reference_images[i], distorted_images[j], pool='mean')
        table[j, 18] = gms(reference_images[i], distorted_images[j], pool='std')

# save MOS value in the fifth column
table[:, 4] = np.fromfile(join(path, 'mos.txt'), sep='\n')

# save MOS standard deviation in the sixth column
table[:, 5] = np.fromfile(join(path, 'mos_std.txt'), sep='\n')

# List with images names
names = []
for image in range(1, 10):
    aux = ['i0{i}_0{d}_{n}.bmp'.format(i=image, d=distortion, n=level)
           for distortion in range(1, 10)
           for level in range(1, 6)] + ['i0{i}_{d}_{n}.bmp'.format(i=image, d=distortion, n=level)
           for distortion in range(10, 25) for level in range(1, 6)]

    names = names + aux

for image in range(10, 26):
    aux = ['i{i}_0{d}_{n}.bmp'.format(i=image, d=distortion, n=level)
           for distortion in range(1, 10)
           for level in range(1, 6)] + ['i{i}_{d}_{n}.bmp'.format(i=image, d=distortion, n=level)
           for distortion in range(10, 25) for level in range(1, 6)]

    names = names + aux

a = tuple([format(x, '.5f') for x in y] for y in table.tolist())  # force it to 5 digits

for x in a[:]:
    x[0] = names[a.index(x)]

# save table to .csv file
with open('../../Data/data.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(headers)
    writer.writerows(a)

# Generate a data set for each distortion type

tables = [np.empty((num_ima * num_lev, 6 + num_indices)) for i in range(1, num_dist + 1)]

r = 0
for subtable in tables:
    r += 1
    nom = []
    for k in range(num_ima * num_lev):
        f = int(k / num_lev) * num_dist_ima + k % num_lev + num_lev * (r - 1)
        subtable[k, ] = table[f, ]
        nom = nom + [names[f]]

    b = tuple([format(x, '.5f') for x in y] for y in subtable.tolist())
    for x in b[:]:
        x[0] = nom[b.index(x)]

    # save the subtables to 24 .csv files
    with open('../../Data/by distortion/distortion{num}.csv'.format(num=r), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(b)
