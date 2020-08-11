"""
Function implementations to calculate correlation and concordance correlation coefficients
"""

import numpy as np
from scipy import stats
import collections
from nltk import agreement


def rank(array):
    """
    Assign a rank to each value and sort them in decreasing order
    :param array: array with the values to sort
    :return: an array with the sorted values
    """
    lis = array.tolist()

    order = array.argsort()

    rank = np.empty((array.shape[0], 1))

    r = 1
    for elem in order:
        rank[elem, ] = r
        r = r + 1

    for elem in lis:
        c = lis.count(elem)
        if c > 1:
            v = 0
            for x in range(c):
                p = lis.index(elem)
                # elements that must be modified are assign the same value
                lis[p] = -10001
                v = v + rank[p, ]
            v = v / c
            for elem in lis:
                if elem == -10001:
                    p = lis.index(elem)
                    # Anything, the important is that it is not repeated
                    # and that it is not 0 or a repeated value of the array
                    lis[p] = -(p + 1000) * v
                    rank[p, ] = v
    return rank


def rho(mos, index):
    """
    Calculate Spearman's correlation between two variables
    :param mos: variable 1
    :param index: variable 2
    :return: coefficient and p-value of bilateral test
    """
    rank_mos = rank(mos)
    rank_index = rank(index)

    dc = (rank_mos - rank_index)**2

    n = mos.shape[0]

    rho = 1 - 6 * sum(dc) / (n * (n**2 - 1))

    # Hypothesis testing:

    # test statistics t >= 0
    t = abs(rho[0]) / (np.sqrt((1 - rho[0]**2) / (n - 2)))

    df = n - 2  # degrees of freedom

    T = stats.t(df)

    # p-value
    p = 2 * (1 - T.cdf(t))  # bilateral test

    return rho[0], p


def con_discon(rank1, rank2):
    """
    Calculates total matching pairs and total mismatching pairs
    :param rank1: array with a certain rank
    :param rank2: array with a certain rank
    :return: Total number of matching pairs and number of mismatching pairs
    """
    n = rank1.shape[0]
    SC = 0
    SD = 0
    for i in range(n):
        C = 0
        D = 0
        for j in range(i + 1, n):
            if (rank1[i] < rank1[j] and rank2[i] < rank2[j]) or (
                    rank1[i] > rank1[j] and rank2[i] > rank2[j]):
                C = C + 1
            elif (rank1[i] < rank1[j] and rank2[i] > rank2[j]) or (rank1[i] > rank1[j] and rank2[i] < rank2[j]):
                D = D + 1
        SC = SC + C
        SD = SD + D
    return SC, SD


def count(rank):
    """
    Counts the number of linked pairs that each group of linked pairs has
    :param rank: groups of linked pairs
    :return: a list g whose elements are the total of linked pairs of each group of linked pairs
    The list size is the number of groups of linked pairs
    """
    g = []
    list_rank = rank.tolist()
    for elem in list_rank:
        i = list_rank.count(elem)
        if i > 1:
            g.append(i)
            for x in range(i):
                list_rank.remove(elem)
    return g


def tauB(mos, index):
    """
    Calculates Kendall's tau correlation between two variables
    :param mos: variable 1
    :param index: variable 2
    :return: coefficient and p-value of bilateral test
    """
    rank_mos = rank(mos)
    rank_index = rank(index)

    C = con_discon(rank_mos, rank_index)[0]
    D = con_discon(rank_mos, rank_index)[1]

    n = mos.shape[0]
    n0 = n * (n - 1) / 2

    t = count(rank_mos)
    u = count(rank_index)

    n1 = 0
    for elem in t:
        n1 = n1 + elem * (elem - 1) / 2

    n2 = 0
    for elem in u:
        n2 = n2 + elem * (elem - 1) / 2

    tau = (C - D) / (np.sqrt((n0 - n1) * (n0 - n2)))

    # Hypothesis testing:

    v0 = n * (n - 1) * (2 * n + 5)

    vt = 0
    for elem in t:
        vt = vt + elem * (elem - 1) * (2 * elem + 5)

    vu = 0
    for elem in u:
        vu = vu + elem * (elem - 1) * (2 * elem + 5)

    v1 = (2 * n1 * 2 * n2) / (2 * n * (n - 1))

    n11 = 0
    for elem in t:
        n11 = n11 + elem * (elem - 1) * (elem - 2)

    n22 = 0
    for elem in u:
        n22 = n22 + elem * (elem - 1) * (elem - 2)

    v2 = (n11 * n22) / (9 * n * (n - 1) * (n - 2))

    v = (v0 - vt - vu) / (18 + v1 + v2)

    zB = abs(C - D) / np.sqrt(v)  # test statistic zB >= 0

    mu = 0
    sigma = 1
    normal = stats.norm(mu, sigma)

    # p-value
    p = 2 * (1 - normal.cdf(zB))  # bilateral test

    return tau, p


def kendall_w(*items):
    """
    Calculates Kendall's w concordance correlation coefficient among a set of variables
    :param items: set of variables
    :return: coefficient and p-value of unilateral test
    """
    ranks = []
    m = 0
    for item in items:
        m = m + 1
        ranks.append(rank(item))

    n = items[0].shape[0]

    Ris = []
    for i in range(n):
        Ri = 0
        for elem in ranks:
            Ri = Ri + elem[i]
        Ris.append(Ri)

    Rbar = sum(Ris) / n

    S = 0
    for Ri in Ris:
        S = S + (Ri - Rbar)**2

    L = 0
    for elem in ranks:
        g = count(elem)
        Lj = 0
        for t in g:
            Lj = Lj + (t**3 - t)
        L = L + Lj

    w = 12 * S / ((m**2) * (n**3 - n) - m * L)

    # Hypothesis testing:

    c2 = m * (n - 1) * w  # test statistic

    df = n - 1  # degrees of freedom

    chi2 = stats.chi2(df)

    # p-value
    p = 1 - chi2.cdf(c2)  # unilateral test
    # p = 1 - stats.chi2.cdf(c2, df) # this would be equivalent

    return w, p


def kappa_pi(mos, index):
    """
    Calculates Cohen's kappa and Scott's pi concordance coefficients between two variables
    :param mos: variable 1
    :param index: variable 2
    :return: kappa coefficient and pi coefficient
    """
    rater0 = mos
    rater1 = index

    n = mos.shape[0]  # 3000

    taskdata_m = []

    for i in range(n):
        taskdata_m.append([0, str(i), str(rater0[i])])  # MOS

    for i in range(n):
        taskdata_m.append([1, str(i), str(rater1[i])])

    ratingtask_m = agreement.AnnotationTask(data=taskdata_m)

    k = ratingtask_m.kappa()

    p = ratingtask_m.pi()

    return k, p


def multi(*items):
    """
    Compute some multivariate concordance indices among a set of variables
    :param items: set of variables
    :return: four multivariate concordance coefficients
    """
    n = items[0].shape[0]  # n = 3000 images

    taskdata = []

    j = 0
    for item in items:
        rater = item.tolist()
        for i in range(n):
            taskdata.append([j, str(i), str(rater[i])])
        j = j + 1

    ratingtask = agreement.AnnotationTask(data=taskdata)

    kappa = ratingtask.kappa()
    fleiss = ratingtask.multi_kappa()
    alpha = ratingtask.alpha()
    scotts = ratingtask.pi()

    return kappa, fleiss, alpha, scotts


def kappa_fleiss(*items):
    """
    Calculate Fleiss' kappa among a set of variables
    :param items: set of variables
    :return: coefficient value
    """
    n = items[0].shape[0]
    # number of items, n = 3000 images

    m = 0
    for item in items:
        m = m + 1
    # number of "evaluators", m = 13 indices

    counter = collections.Counter(items[0])
    c = len(counter)
    # There are c = 3 categories

    N = np.empty((n, c))

    # N[i,j] it's the number of evaluators that assigned
    # the j-th category to the i-th item, j = 1,...,c

    data = np.empty((n, m))

    j = 0
    for item in items:
        data[:, j] = item
        j = j + 1

    for i in range(n):
        contador = collections.Counter(data[i, :])
        for j in range(c):
            N[i, j] = contador[j + 1]

    # This would be equivalent:
    #    for i in xrange(n):
    #        for j in xrange(c):
    #            l = data[i,:].tolist()
    #            N[i,j] = l.count(j+1)

    p = []
    for x in range(c):
        pi = sum(N[:, x]) / (n * m)
        p.append(pi)

    # notice that p1 + p2 + p3 = 1

    Pe = 0
    for elem in p:
        Pe = Pe + elem**2

    # Pe = p1**2 + p2**2 + p3**2

    # sum the squares of the elements of matrix N
    S = 0
    for i in range(n):
        for j in range(c):
            S = S + N[i, j]**2

    Pbar = (S - n * m) / (n * m * (m - 1))

    KF = (Pbar - Pe) / (1 - Pe)

    return KF
