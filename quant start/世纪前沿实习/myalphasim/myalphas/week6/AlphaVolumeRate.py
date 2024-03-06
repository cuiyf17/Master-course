# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pickle
from alphasim import AlphaBase
from alphasim import DataRegistry as dr
from alphasim import Oputil
from alphasim import Universe as uv

# 将其他路径添加到sys.path中
sys.path.append(os.path.abspath('/home/cuiyf/myalphasim/'))
import cuiyf_op.cuiyfOp as OP

from sklearn.mixture import GaussianMixture
class Alpha(AlphaBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ndays = cfg.getAttributeDefault('ndays', 21)
        self.lag = cfg.getAttributeDefault('lag', 0)
        self.cap = dr.getData('cap')
        self.close = dr.getData('adj_close')
        self.status = dr.getData('status')
        self.volume = dr.getData('adj_volume')
        self.subindustry = dr.getData('subindustry')
        self.risk_free_return = 0
        self.pre_alpha = None
        self.std_returns1 = None
        self.std_returns2 = None
        self.KK = None


    def generate(self, di):
        start_di = di - self.delay - self.lag - self.ndays + 1
        end_di = di - self.delay - self.lag + 1

        valid_idx = self.valid[di]#& self.valid[di-1] & (self.volume[di - 1] > 0) & (self.volume[di] > 0)
        #停牌后复牌的股票不要
        valid_idx = valid_idx & ~((self.status[di - 1] == 0) & (self.status[di] == 1))

        subindustry = self.subindustry[di-self.delay, valid_idx]
        g_returns = self.close[di-self.delay, valid_idx]/self.close[di-self.delay-21, valid_idx] - 1
        g_close = self.close[di-self.delay, valid_idx]
        group1 = OP.group_split2(g_returns, 10)
        group2 = OP.group_split2(g_close, 10)

        prices = self.volume[start_di:end_di, valid_idx] / self.volume[start_di-5:end_di-5, valid_idx] - 1
        prices[np.isinf(prices)] = np.nan
        prices = prices - np.nanmean(prices, axis=1, keepdims=True)
        prices1 = self.volume[start_di-1:end_di-1, valid_idx] / self.volume[start_di-6:end_di-6, valid_idx] - 1
        prices1[np.isinf(prices1)] = np.nan
        prices1 = prices1 - np.nanmean(prices1, axis=1, keepdims=True)
        
        price0 = prices[0]
        ER = (np.abs(prices - price0)/np.nansum(np.abs(prices - prices1), axis=0))[-1]
        sma1 = 2/(2+1)
        sma2 = 2/(30+1)
        SC = (ER*(sma2-sma1)+sma1)**2

        tmp = np.zeros(self.close[di-self.delay].shape)
        tmp[valid_idx] = prices[-1]
        if self.KK is None:
            self.KK = tmp.copy()
        else:
            #mask = np.isnan(tmp[valid_idx])
            #self.KK[valid_idx][mask] = SC[mask]*tmp[valid_idx][mask] + (1-SC[mask])*self.KK[valid_idx][mask]
            self.KK[valid_idx] = SC*tmp[valid_idx] + (1-SC)*self.KK[valid_idx]

        alpha = -prices[-1]
        
        # alpha = OP.groupNeutralize(alpha, group1)
        # alpha = OP.groupNeutralize(alpha, group2, standardization = False)
        

        if(self.pre_alpha is None):
            self.alpha[valid_idx] = alpha
            self.pre_alpha = self.alpha.copy()
        else:
            mask1 = ~np.isnan(self.pre_alpha[valid_idx])
            mask2 = ~np.isnan(self.pre_alpha)
            if((di - self.delay)%1 == 0):
                self.alpha[valid_idx&mask2] = 0.9*self.pre_alpha[valid_idx&mask2] + 0.1*alpha[mask1]
                self.alpha[valid_idx&~mask2] = alpha[~mask1]
                self.pre_alpha = self.alpha.copy()
            else:
                self.alpha[valid_idx&mask2] = 1*self.pre_alpha[valid_idx&mask2] + 0*alpha[mask1]
                self.alpha[valid_idx&~mask2] = alpha[~mask1]
                self.pre_alpha = self.alpha.copy()

    def checkpointSave(self, fh):
        pickle.dump(self.pre_alpha, fh)
        pickle.dump(self.KK, fh)


    def checkpointLoad(self, fh):
        self.pre_alpha = pickle.load(fh)
        self.KK = pickle.load(fh)
