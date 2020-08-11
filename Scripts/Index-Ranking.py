"""
Sort quality indices according to each concordance and correlation coefficient
Then generate a final quality index ranking based on these single rankings
"""

import pandas as pd
from collections import OrderedDict


def sort_by_one_coeff(table, file_name, figures=4):
    """
    Given a table with a correlation (or concordance) value between each quality index and the MOS, for each distortion
    group, sort the indices in descending order, from highest to lowest correlation (concordance), generating a quality
    index ranking for each distortion group according to that correlation (concordance) coefficient
    :param table: pandas data frame with the correlation or concordance coefficients values
    :param file_name: path to save the quality index ranking to a csv file
    :param figures: decimal figures to consider
    :return: pandas data frame with the quality index ranking
    """
    data = table.abs().round(figures)

    d = OrderedDict()

    for col in data:
        sorted_data = pd.DataFrame.sort_values(data, by=col, ascending=False)
        d.update({col: list(sorted_data[col].index)})

    frame = pd.DataFrame(d)
    frame.to_csv(file_name)

    return frame


def final_ranking(*items, file_name, n=12):
    """
    Given a set of index rankings based on a single coefficient, generates a final index ranking that summarizes them
    all. The final ranking only has three positions: first, second and third best quality index for each distortion
    group. The way to find them is as follows: for each single ranking add 3, 2 and 1 point to the first, second and
    third index respectively. Then the index with higher score is in first place, the one that follows second and
    the one that follows third. There may be ties.
    :param items: set of index rankings based on a single coefficient, each one must be a pandas data frame
    :param file_name: path to save the quality index ranking to a csv file
    :param n: amount of quality indices to compare
    :return: pandas data frame with the quality index ranking
    """

    d = OrderedDict()

    groups = list(items[0].columns)

    for g in groups:

        points = {}

        indices = ['MSE', 'PSNR', 'SNR', 'WSNR', 'NQM', 'UQI', 'SSIM', 'MSSIM', 'VIF', 'CQ(1,1)', 'GMSM', 'GMSD']

        for key in indices:
            points.update({key: 0})

        for item in items:

            first = item[g][0]
            second = item[g][1]
            third = item[g][2]
            fourth = item[g][3]
            fifth = item[g][4]
            sixth = item[g][5]
            seventh = item[g][6]
            eighth = item[g][7]
            ninth = item[g][8]
            tenth = item[g][9]
            eleventh = item[g][10]
            twelfth = item[g][11]

            points.update({first: points[first] + n - 1})
            points.update({second: points[second] + n - 2})
            points.update({third: points[third] + n - 3})
            points.update({fourth: points[fourth] + n - 4})
            points.update({fifth: points[fifth] + n - 5})
            points.update({sixth: points[sixth] + n - 6})
            points.update({seventh: points[seventh] + n - 7})
            points.update({eighth: points[eighth] + n - 8})
            points.update({ninth: points[ninth] + n - 9})
            points.update({tenth: points[tenth] + n - 10})
            points.update({eleventh: points[eleventh] + n - 11})
            points.update({twelfth: points[twelfth] + n - 12})

        for key in points:
            points.update({key: [points[key]]})

        frame = pd.DataFrame.from_dict(points)
        sorted_data = pd.DataFrame.sort_values(frame, by=0, axis=1, ascending=False)
        sorted_data['--'] = -1

        cols = list(sorted_data.columns)

        if len(cols) == 12:
            rank = cols
        else:
            rank = []

            for i in range(len(cols)-1):

                if sorted_data[cols[i]][0] == sorted_data[cols[i+1]][0]:
                    rank.append(cols[i]+'-'+cols[i+1])
                    cols.append('--')
                    del cols[i+1]
                else:
                    rank.append(cols[i])

        d.update({g: rank})

    frame = pd.DataFrame(d)
    frame.to_csv(file_name)
    return frame


table1 = pd.read_csv('../Results/Spearman-correlation.csv', index_col=0)
frame1 = sort_by_one_coeff(table1, file_name='Results/Sort-by-Spearman.csv')

table2 = pd.read_csv('../Results/Kendall-correlation.csv', index_col=0)
frame2 = sort_by_one_coeff(table2, file_name='Results/Sort-by-Kendall.csv')

table3 = pd.read_csv('../Results/Cohen-concordance.csv', index_col=0)
frame3 = sort_by_one_coeff(table3, file_name='Results/Sort-by-Cohen.csv')

table4 = pd.read_csv('../Results/Scott-concordance.csv', index_col=0)
frame4 = sort_by_one_coeff(table4, file_name='Results/Sort-by-Scott.csv')

final_ranking(frame1, frame2, frame3, frame4, file_name='../Results/Index-ranking.csv')
