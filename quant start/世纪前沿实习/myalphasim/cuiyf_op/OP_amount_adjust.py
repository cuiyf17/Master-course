from alphasim import AlphaOpBase
from alphasim import Oputil
from alphasim import DataRegistry as dr
from alphasim.data import Universe as uv
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pdb

def industrydemean(x, g, me=2):
    groups = np.unique(g)
    for k in groups:
        idx = (g == k) & ~np.isnan(x)
        if k < 0 or np.sum(idx) < me:
            x[idx] = np.nan
            continue
        x[idx] = x[idx] - np.mean(x[idx])
def DECAYLINEAR(A):
    inds = A.shape[0]
    cols = A.shape[1]
    w=np.arange(1,inds+1).repeat(cols).reshape((inds,cols))
    w = np.where(~np.isnan(A),w,np.nan)
    w=w/np.nansum(w,axis=0)
    return np.nansum(A*w,axis=0)

def EWM_lt(A,halflife):
    beta = (1/2)**(1/halflife)
    inds = A.shape[0]
    cols = A.shape[1]
    W = np.zeros(inds)
    for i in range(inds):
        W[-(i+1)] = beta**(i) 
    w=W.repeat(cols).reshape((inds,cols))
    w = np.where(~np.isnan(A),w,np.nan)
    w=w/np.nansum(w,axis=0)
    return np.nansum(A*w,axis=0)

def get_ts_ma(mat, window, ratio=None):
    window = np.int(np.floor(window))
    (m, n) = mat.shape
    ans = np.full((m, n), np.nan)

    if not ratio:
        for i in range(window - 1, m):
            ans[i, :] = np.nanmean(mat[i - window + 1: i + 1, :], axis=0)
    else:
        least = np.int(np.floor(window * ratio))
        flag = np.double(~np.isnan(mat))

        for i in range(window - 1, m):
            temp = np.nanmean(mat[i - window + 1: i + 1, :], axis=0)
            temp_filter = np.nansum(flag[i - window + 1: i + 1, :], axis=0) < least
            temp[temp_filter] = np.nan
            ans[i, :] = temp

    return ans

def get_ts_max(mat, window):
    window = np.int(np.floor(window))
    (m, n) = mat.shape
    ans = np.full((m, n), np.nan)

    for i in range(window - 1, m):
        ans[i, :] = np.nanmax(mat[i - window + 1: i + 1, :], axis=0)

    return ans

class amount_adjust(AlphaOpBase):
    def __init__(self, cfg):
        AlphaOpBase.__init__(self, cfg)
        self.SS0 = dr.getData('TOP2000')
        self.amount = dr.getData('amount')
        self.open = dr.getData('adj_open')
        self.close = dr.getData('adj_close')
        self.vwap = dr.getData('adj_vwap')
        self.high = dr.getData('adj_high')
        self.low = dr.getData('adj_low')
        self.negcap = dr.getData('negcap')
        self.iclose = dr.getData('interval5m.close')
        self.iopen = dr.getData('interval5m.open')
        #self.iamount = dr.getData('interval5m.amo')
        #self.industry = dr.getData('industry')
        #self.SS2 = dr.getData('ZZ500')
        
        return

    def apply(self, di, alpha):
        
        SS0 = np.copy(self.SS0[di-1])
        #high = np.copy(self.high[di-20:di]) 
        #close = np.copy(self.close[di-41:di])
        #cap = np.copy(self.cap[di-20:di])
        #aig = np.nanmean(cap,axis=0)
        #ret = close[1:] / close[:-1] - 1
        #for i in range(len(ret)):
        #    ret[i][~SS0] = np.nan
        #ft = np.nanstd(ret,axis=1)
        #ft = np.nanmean(ft[-5:]) / np.nanmean(ft)
        #ft = np.nanmean(ft)
        #pdb.set_trace()
        #ft = np.where(ft<1.2,np.where(ft>0.8,ft,0.8),1.2)
        #pdb.set_trace()
        #rg = (np.copy(self.high[di-20:di]) / np.copy(self.low[di-20:di]) - 1)
        
        #aig =  np.nanquantile(rg,0.5,axis=0)
        #aig[~SS0] = np.nan
        #adj1 = aig#**0.5

        high = np.copy(self.high[di-80:di])
        low = np.copy(self.low[di-80:di])
        rg = high / low - 1

        close = np.copy(self.close[di-80:di])
        open = np.copy(self.open[di-80:di])
        aor = np.abs(close / open -1) 
        #aig = np.nanmean(high,axis=0) / np.nanmean(low,axis=0) - 1 
        #aig1 = np.nanmedian(rg[-20:],axis=0)
        #sig = rg / get_ts_max(rg,10)
        #sig = sig[-80:]
        rg = rg - aor
        aig1 = np.nanmean(rg,axis=0)
        aig2 = np.nanmax(rg,axis=0)/2 + np.nanmin(rg,axis=0)/2
        #aig3 = np.nanmean(rg,axis=0)/2
        
        aig1[np.isnan(alpha)] = np.nan
        aig2[np.isnan(alpha)] = np.nan
        #aig3[np.isnan(alpha)] = np.nan
        #close = np.copy(self.close[di-21:di])
        #rg = close[1:] / close[:-1] - 1
        #aig1 = np.nanstd(rg,axis=0)
        #aig1[np.isnan(alpha)] = np.nan

        aig = aig1 / (aig2)
        #aig = aig

        #aig =  np.nanquantile(rg,0.5,axis=0)
        #aig[~SS0] = np.nan
        aig[np.isnan(alpha)] = np.nan
        adj2 = aig# **0.5

        tr = np.copy(self.amount[di-242:di]) / np.copy(self.negcap[di-242:di])
        aig = np.nanmean(tr,axis=0)
        aig[np.isnan(alpha)] = np.nan
        aig = aig / np.nanquantile(aig,0.5)
        aig = np.where(aig>1,aig,np.where(~np.isnan(aig),1,np.nan))
        #aig =  np.nanquantile(rg,0.5,axis=0)
        aig[~SS0] = np.nan
        adj3 = (aig)**(0.3)

        amount = np.copy(self.amount[di-20:di])
        aig = np.nanmean(amount,axis=0)
        #aig = aig - np.nanquantile(aig,0.5)
        #aig = np.abs(aig)
        
        #aig =  np.nanquantile(rg,0.5,axis=0)
        
        aig[~SS0] = np.nan
        #pdb.set_trace()
        aig[np.isnan(alpha)] = np.nan
        #aig = aig - np.nanquantile(aig,0.5)
        #aig = np.abs(aig)
        adj4 = aig**(0.5)
        #adj = adj3 / np.nansum(adj3) + adj2 / np.nansum(adj2) + adj4 / np.nansum(adj4)
        adj = adj4 * adj2# * adj3

        alpha[:] = alpha * (adj)
        
        return