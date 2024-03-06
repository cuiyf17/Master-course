#!/home/cuiyf/anaconda3/bin/python3
from scipy.stats import rankdata as rd
from scipy.stats.stats import pearsonr as pr
from scipy.stats.stats import skew as sskew
from scipy.stats.stats import kurtosis as skurtosis
import numpy as np
import math
import scipy

def power(X, exp, dorank=False):
    x = X.copy()
    if dorank:
        x = rank(x)
        x -= 0.5
    x[:] = np.sign(x) * np.power(np.fabs(x), exp)
    return x

def rank(X, method='average', axis = None):  # min/max or average, do not use dense (not symmetric around 0.5)
    # iimap = np.nonzero(np.logical_not(np.isnan(x)))[0]
    x = X.copy()
    iimap = ~np.isnan(x)
    y = x[iimap]
    n = len(y)
    if n == 0:
        return x
    elif n == 1:
        y[0] = 0.5
    else:
        y = (rd(y, method, axis=axis) - 1) / float(n - 1)
    x[iimap] = y[:]
    return x

def tsrank(X, method='average'):
    # min/max or average, do not use dense (not symmetric around 0.5)
    x = X.copy()
    for i in range(X.shape[1]):
        iimap = ~np.isnan(x[:,i])
        y = x[iimap, i]
        n = len(y)
        if n == 0:
            continue
        elif n == 1:
            y[0] = 0.5
        else:
            y = (rd(y, method) - 1) / float(n - 1)
        x[iimap, i] = y[:]
    return x

def sum(x, me=1, axis=0):
    r = np.nansum(x, axis=axis)
    n = np.sum(~np.isnan(x), axis=axis)
    if x.ndim == 1:
        if n < me:
            r = np.nan
    else:
        r[n < me] = np.nan
    return r

def mean(x, me=1, axis=0):
    r = np.nanmean(x, axis=axis)
    n = np.sum(~np.isnan(x), axis=axis)
    if x.ndim == 1:
        if n < me:
            r = np.nan
    else:
        r[n < me] = np.nan
    return r


def median(x, me=1, axis=0):
    r = np.nanmedian(x, axis=axis)
    n = np.sum(~np.isnan(x), axis=axis)
    if x.ndim == 1:
        if n < me:
            r = np.nan
    else:
        r[n < me] = np.nan
    return r


def std(x, me=2, axis=0):
    r = np.nanstd(x, axis=axis)
    n = np.sum(~np.isnan(x), axis=axis)
    if x.ndim == 1:
        if n < me:
            r = np.nan
    else:
        r[n < me] = np.nan
    return r


def skew(x, me=3, axis=0):
    r = sskew(x, axis=axis, nan_policy='omit')
    n = np.sum(~np.isnan(x), axis=axis)
    if x.ndim == 1:
        if n < me:
            r = np.nan
    else:
        r[n < me] = np.nan
    return r


def kurt(x, me=4, axis=0):
    r = skurtosis(x, axis=axis, nan_policy='omit')
    n = np.sum(~np.isnan(x), axis=axis)
    if x.ndim == 1:
        if n < me:
            r = np.nan
    else:
        r[n < me] = np.nan
    return r

def moment(x, moment = 1, me=4, axis=0):
    mean = np.nanmean(x, axis=axis)
    std = np.nanstd(x, axis=axis)
    r = np.nanmean(np.power((x - mean)/std, moment), axis=axis)
    n = np.sum(~np.isnan(x), axis=axis)
    if x.ndim == 1:
        if n < me:
            r = np.nan
    else:
        r[n < me] = np.nan
    return r

# correlation
def corr_vector(x, y, me=3):
    valid = (~np.isnan(x)) & (~np.isnan(y))
    if np.sum(valid) < me:
        return np.nan
    r, p = pr(x[valid], y[valid])
    return r



# correlation of two matrix, column by column
def corr(x, y, me=3, axis=0):
    if x.shape != y.shape:
        raise ValueError('shape mismatch: x=%s y=%s' % (str(x.shape), str(y.shape)))
    x1 = np.copy(x)
    y1 = np.copy(y)
    idx = np.isnan(x1) | np.isnan(y1)
    x1[idx] = np.nan
    y1[idx] = np.nan
    xy = x1 * y1
    xx = x1 * x1
    yy = y1 * y1
    sx = np.nansum(x1, axis=axis)
    sy = np.nansum(y1, axis=axis)
    sxy = np.nansum(xy, axis=axis)
    sxx = np.nansum(xx, axis=axis)
    syy = np.nansum(yy, axis=axis)
    n = np.sum((~np.isnan(x1)) & (~np.isnan(y1)), axis=axis)
    r = (n * sxy - sx * sy) / np.sqrt(n * sxx - sx * sx) / np.sqrt(n * syy - sy * sy)
    if x.ndim == 1:
        # r = max(min(r, 1.0), -1.0)
        r = np.maximum(np.minimum(r, 1.0), -1.0)
        if n < me:
            r = np.nan
    else:
        # r[ r > 1.0 ] = 1.0
        # r[ r < -1.0 ] = -1.0
        r = np.maximum(np.minimum(r, 1.0), -1.0)
        r[n < me] = np.nan
    return r


# covariance of two matrix, column by column
def cov(x, y, me=3, axis=0):
    if x.shape != y.shape:
        raise ValueError('shape mismatch: x=%s y=%s' % (str(x.shape), str(y.shape)))
    x1 = np.copy(x)
    y1 = np.copy(y)
    idx = np.isnan(x1) | np.isnan(y1)
    x1[idx] = np.nan
    y1[idx] = np.nan
    xy = x1 * y1
    sx = np.nansum(x1, axis=axis)
    sy = np.nansum(y1, axis=axis)
    sxy = np.nansum(xy, axis=axis)
    n = np.sum((~np.isnan(x1)) & (~np.isnan(y1)), axis=axis)
    r = (n * sxy - sx * sy) / n / np.maximum(n - 1, 1)
    if x.ndim == 1:
        if n < me:
            r = np.nan
    else:
        r[n < me] = np.nan
    return r


# coskewness (???)
def coskew(x, y, me=5, axis=0):
    if x.shape != y.shape:
        raise ValueError('shape mismatch: x=%s y=%s' % (str(x.shape), str(y.shape)))
    x1 = np.copy(x)
    y1 = np.copy(y)
    idx = np.isnan(x1) | np.isnan(y1)
    x1[idx] = np.nan
    y1[idx] = np.nan
    mx = np.nanmean(x1, axis=axis)
    my = np.nanmean(y1, axis=axis)
    xxy = (x1 - mx) * (x1 - mx) * (y1 - my)
    sxxy = np.nansum(xxy, axis=axis) / np.sum(~np.isnan(xxy), axis=axis)
    xx = (x1 - mx) * (x1 - mx)
    sxx = np.nansum(xx, axis=axis) / np.sum(~np.isnan(xx), axis=axis)
    yy = (y1 - my) * (y1 - my)
    syy = np.nansum(yy, axis=axis) / np.sum(~np.isnan(yy), axis=axis)
    cs = sxxy / sxx / np.sqrt(syy)
    n = np.sum((~np.isnan(x1)) & (~np.isnan(y1)), axis=axis)
    if x.ndim == 1:
        if n < me:
            cs = np.nan
    else:
        cs[n < me] = np.nan
    return cs


# cokurtosis (???)
def cokurt(x, y, me=6, axis=0):
    if x.shape != y.shape:
        raise ValueError('shape mismatch: x=%s y=%s' % (str(x.shape), str(y.shape)))
    x1 = np.copy(x)
    y1 = np.copy(y)
    idx = np.isnan(x1) | np.isnan(y1)
    x1[idx] = np.nan
    y1[idx] = np.nan
    mx = np.nanmean(x1, axis=axis)
    my = np.nanmean(y1, axis=axis)
    xxyy = (x1 - mx) * (x1 - mx) * (y1 - my) * (y1 - my)
    sxxyy = np.nansum(xxyy, axis=axis) / np.sum(~np.isnan(xxyy), axis=axis)
    xx = (x1 - mx) * (x1 - mx)
    sxx = np.nansum(xx, axis=axis) / np.sum(~np.isnan(xx), axis=axis)
    yy = (y1 - my) * (y1 - my)
    syy = np.nansum(yy, axis=axis) / np.sum(~np.isnan(yy), axis=axis)
    ck = sxxyy / sxx / syy
    n = np.sum((~np.isnan(x1)) & (~np.isnan(y1)), axis=axis)
    if x.ndim == 1:
        if n < me:
            ck = np.nan
    else:
        ck[n < me] = np.nan
    return ck



# =========================================================================================================================================================================

def neutralizeVector(input, vector, method='Schmidt'):
    try:
        input.shape == vector.shape
    except Exception as e:
        raise e
    if(method == 'Schmidt'):
        valid = (~np.isnan(input)) & (~np.isnan(vector))
        x = input[valid]
        y = vector[valid]
        beta = np.inner(x, y) / np.inner(y, y)
        ret = input - vector * beta
    elif(method == "LinearRegression"):
        valid = (~np.isnan(input)) & (~np.isnan(vector))
        y = input[valid]
        x = vector[valid]
        #x = np.concatenate([np.ones(1), x])
        beta = (x.T @ y)/(x.T @ x)
        alpha = np.nanmean(y - beta * x)
        ret = input - vector * beta - alpha
    return ret

def ts_neutralize_vector(input, vector, method='Schmidt'):
    try:
        input.shape == vector.shape
    except Exception as e:
        raise e
    ret = np.zeros(input.shape)
    for i in range(input.shape[1]):
        valid = (~np.isnan(input[:,i])) & (~np.isnan(vector[:,i]))
        x = input[valid,i]
        y = vector[valid,i]
        beta = np.inner(x, y) / np.inner(y, y)
        ret[:,i] = input[:,i] - vector[:,i] * beta
    return ret

def group_split(x, n_group):
    """
    :param x: 1d array
    :param n_group: int
    :return: 1d array
    """
    try:
        x.shape[0] > n_group
    except Exception as e:
        raise e
    ret = np.zeros(x.shape[0])
    idx = np.argsort(x)
    for i in range(n_group):
        ret[idx[i::n_group]] = i
    return ret

def group_split2(x, n_group):
    """
    :param x: 1d array
    :param n_group: int
    :return: 1d array
    """
    try:
        x.shape[0] > n_group
    except Exception as e:
        raise e
    ret = np.zeros(x.shape[0])
    idx = np.argsort(x)
    group_size = math.ceil(x.shape[0] / n_group)
    for i in range(n_group):
        ret[idx[i*group_size:min((i+1)*group_size, x.shape[0])]] = i
    return ret

def quantileTruncate(X, quantile = 0.05, set_nan = False):
    """
    :param x: 1d array
    :param quantile: float
    :return: 1d array
    """
    x = X.copy()
    valid = ~np.isnan(x)
    upper = np.percentile(x[valid], 100 - quantile * 100)
    lower = np.percentile(x[valid], quantile * 100)
    if(not set_nan):
        ret = np.where(x[valid] > upper, upper, x[valid])
        ret = np.where(ret < lower, lower, ret)
    else:
        ret = np.where(x[valid] > upper, np.nan, x[valid])
        ret = np.where(ret < lower, np.nan, ret)
    x[valid] = ret
    return x

def groupRank(X, g, method="average", me=2):
    x = X.copy()
    groups = np.unique(g)
    tot_stock = np.sum((g >= 0) & ~np.isnan(x))
    for k in groups:  # do rank within each group
        if k < 0:
            x[g == k] = np.nan  # 20170721
            continue
        idx = (~np.isnan(x)) & (g == k)
        cnt = np.sum(idx)
        if cnt < me:
            x[idx] = np.nan
            continue
        y = x[idx]  # temp array to store elements of group k
        n = len(y)
        if n == 0:
            return
        elif n == 1:
            y[0] = 0.5
        else:
            y = (rd(y, method) - 1) / float(n - 1)
        x[idx] = y  # 20171122: remove normalization
    return x

def groupNeutralize(X, g, me=2, standardization=True):
    x = X.copy()
    groups = np.unique(g)
    for k in groups:
        idx = (g == k) & ~np.isnan(x)
        if k < 0 or np.sum(idx) < me:
            x[idx] = np.nan
            continue
        x[idx] -= np.mean(x[idx])
        if standardization:
            x[idx] /= np.std(x[idx])
    return x

def group_mean(X, g, me=2, skipnan = True):
    x = X.copy()
    groups = np.unique(g)
    if(skipnan):
        for k in groups:
            idx = (g == k) & ~np.isnan(x)
            if k < 0 or np.sum(idx) < me:
                x[idx] = np.nan
                continue
            x[idx] = np.mean(x[idx])
    else:
        for k in groups:
            idx = (g == k)
            if k < 0 or np.sum(idx) < me:
                x[idx] = np.nan
                continue
            x[idx] = np.nanmean(x[idx])
    return x

def group_max(X, g, me=2, skipnan = True):
    x = X.copy()
    groups = np.unique(g)
    if(skipnan):
        for k in groups:
            idx = (g == k) & ~np.isnan(x)
            if k < 0 or np.sum(idx) < me:
                x[idx] = np.nan
                continue
            x[idx] = np.max(x[idx])
    else:
        for k in groups:
            idx = (g == k)
            if k < 0 or np.sum(idx) < me:
                x[idx] = np.nan
                continue
            x[idx] = np.nanmax(x[idx])
    return x

def group_min(X, g, me=2, skipnan = True):
    x = X.copy()
    groups = np.unique(g)
    if(skipnan):
        for k in groups:
            idx = (g == k) & ~np.isnan(x)
            if k < 0 or np.sum(idx) < me:
                x[idx] = np.nan
                continue
            x[idx] = np.min(x[idx])
    else:
        for k in groups:
            idx = (g == k)
            if k < 0 or np.sum(idx) < me:
                x[idx] = np.nan
                continue
            x[idx] = np.nanmin(x[idx])
    return x

def groupZscore(x, g, me=3):
    groups = np.unique(g)
    for k in groups:
        idx = (g == k) & ~np.isnan(x)
        if k < 0 or np.sum(idx) < me:
            x[idx] = np.nan
            continue
        x[idx] = (x[idx] - mean(x[idx], me = me)) / std(x, me = me)

def mean(x, me=1, axis=0):
    r = np.nanmean(x, axis=axis)
    n = np.sum(~np.isnan(x), axis=axis)
    if x.ndim == 1:
        if n < me:
            r = np.nan
    else:
        r[n < me] = np.nan
    return r

def tsRSI(x, axis = 0):
    delta = np.diff(x, axis = axis)
    up_mask = (delta > 0) & ~np.isnan(delta)
    down_mask = delta <= 0 & ~np.isnan(delta)
    up = np.nansum(delta * up_mask, axis = axis)
    down = np.nansum(-delta * down_mask, axis = axis)
    rsi = up / (up + down)
    return rsi

def tsEMA(x, me=1, axis=0, alpha = None):
    N = len(x)
    if(alpha is None):
        alpha = 2/(N + 1)
    n = np.sum(~np.isnan(x), axis=axis)
    weights = np.ones_like(x) * (np.power(1 - alpha, np.arange(N-1, -1, -1)).reshape(-1, 1))
    valid = ~np.isnan(x)
    tmp_x = x * weights
    tmp_x[~valid] = 0
    weights[~valid] = 0
    r = np.nansum(tmp_x, axis=axis) / np.nansum(weights, axis=axis)
    if x.ndim == 1:
        if n < me:
            r = np.nan
    else:
        r[n < me] = np.nan
    return r

def tsDFT1(x, axis = 0):
    N = len(x)
    w = np.exp(-2j * np.pi * np.arange(N) / N).repeat(x.shape[1]).reshape(-1, x.shape[1])
    r = np.fft.fft(x * w, axis = axis)
    return r

def tsDFT2(x, axis = 0, degree = 0, RorI = "R"):
    N = len(x)
    mask = None
    if(RorI == "R"):
        mask = (np.cos(2*np.pi*np.arange(N - 1, -1, -1)/N*degree)).reshape(-1, 1)
    elif(RorI == "I"):
        mask = (np.sin(-2*np.pi*np.arange(N - 1, -1, -1)/N*degree)).reshape(-1, 1)
    else:
        raise ValueError
    result = np.nansum(mask*x, axis=axis)
    return result


def std(x, me=2, axis=0):
    r = np.nanstd(x, axis=axis)
    n = np.sum(~np.isnan(x), axis=axis)
    if x.ndim == 1:
        if n < me:
            r = np.nan
    else:
        r[n < me] = np.nan
    return r

def zscore(x, me = 3, axis = 0):
    """
    :param x: 1d array
    :param me: int
    :return: 1d array
    """
    r = (x - mean(x, me = me, axis = axis)) / std(x, me = me, axis = axis)
    return r

def truncate(x, maxPercent=0.1, maxIter=1):
    update = 0
    for i in range(maxIter):
        sum_l = np.sum(x[x > 0])
        sum_s = np.sum(x[x < 0])
        # print sum_l, sum_s, np.max(x[~np.isnan(x)]), np.min(x[~np.isnan(x)])
        idx = x > sum_l * maxPercent
        x[idx] = sum_l * maxPercent
        update += np.sum(idx)
        idx = x < sum_s * maxPercent
        x[idx] = sum_s * maxPercent
        update += np.sum(idx)
        if update == 0:
            return 0
        if i == maxIter - 1 and update > 0:
            print('*** warning ***: %d extreme value in last iter in truncate!' % update)
            return -1
    return 0

def residual(x, inputs):
    '''
    get residuals of x ~ a * inputs + b
    params
    inputs: a vector or a list of vectors
    '''
    # if not isinstance(factors, list):
    #    raise ValueError('residual: input should be a list of vectors')
    if isinstance(inputs, list):
        factors = inputs
    else:
        factors = [inputs]
    valid = (~np.isnan(x))
    for f in factors:
        valid &= (~np.isnan(f))
    x_valid = x[valid]
    fac_valid = []
    for f in factors:
        fac_valid.append(f[valid])
    fac_valid.append(np.ones(len(x_valid)))
    fac_valid = np.vstack(fac_valid).T
    r = np.linalg.lstsq(fac_valid, x_valid)[0]
    x[~valid] = np.nan
    x[valid] -= fac_valid.dot(r.T)

def tsLinearRegression(X, Y, output = "beta"):
    """
    :param X: 2d array, n_samples * n_features
    :param Y: 1d array, n_samples
    :return: 1d array, n_features
    """
    try:
        X.shape[0] == Y.shape[0]
    except Exception as e:
        raise e
    
    valid_idx = None
    N, M = X.shape
    for i in range(M):
        if valid_idx is None:
            valid_idx = ~np.isnan(X[:, i])
        else:
            valid_idx = valid_idx & ~np.isnan(X[:, i])
    valid_idy = ~np.isnan(Y)

    beta = []
    alpha = []
    residual = []

    for i in range(Y.shape[1]):
        valid = valid_idx & valid_idy[:, i]
        x = X[valid]
        ones = np.ones((x.shape[0], 1))
        x = np.concatenate([ones, x], axis = 1)
        tmp = np.linalg.inv(x.T @ x) @ x.T @ (Y[valid, i])
        if output == "beta":
            beta.append(tmp[1:])
        elif output == "alpha":
            alpha.append(tmp[0])
        elif output == "residual":
            residual.append(Y[valid, i] - x @ tmp)
        else:
            beta.append(tmp[1:])
            alpha.append(tmp[0])
            residual.append(Y[valid, i] - x @ tmp)
    if output == "beta":
            ret = np.concatenate(beta)
    elif output == "alpha":
            ret = np.array(alpha)
    elif output == "residual":
            ret = np.array(residual).T
    else:
        beta = np.concatenate(beta)
        alpha = np.array(alpha)
        residual = np.array(residual).T
    
        ret = (alpha, beta, residual)

    return ret

class KalmanFilter():
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)