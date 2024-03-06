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


class Alpha(AlphaBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ndays = cfg.getAttributeDefault('ndays', 21)
        self.lag = cfg.getAttributeDefault('lag', 0)
        self.cap = dr.getData('cap')
        self.close = dr.getData('close')
        self.adj_close = dr.getData('adj_close')
        self.status = dr.getData('status')
        self.volume = dr.getData('adj_volume')
        self.factor = dr.getData('adjfactor')
        self.tclose = dr.getData('close')
        self.risk_free_return = 0
        self.pre_alpha = None
        self.std_returns1 = None
        self.std_returns2 = None
        self.start_price = None

    def generate(self, di):
        start_di = di - self.delay - self.lag - self.ndays + 1
        end_di = di - self.delay - self.lag + 1

        valid_idx = self.valid[di] #& self.valid[di-1] & (self.volume[di - 1] > 0) & (self.volume[di] > 0)
        #停牌后复牌的股票不要
        valid_idx = valid_idx & ~((self.status[di - 1] == 0) & (self.status[di] == 1))

        if(self.start_price is None):
            tmp = np.zeros(self.close[di].shape[0])
            tmp[valid_idx] = self.close[di-self.delay, valid_idx]*self.factor[di-self.delay, valid_idx]
            self.start_price = tmp.copy()
        else:
            tmp = np.zeros(self.close[di].shape[0])
            tmp[valid_idx] = self.close[di-self.delay, valid_idx]
            mask = np.isnan(self.start_price)
            self.start_price[mask&valid_idx] = tmp[mask&valid_idx]*self.factor[di-self.delay, mask&valid_idx]
            self.start_price[(~mask)&valid_idx] = self.start_price[(~mask)&valid_idx]*self.factor[di-self.delay, (~mask)&valid_idx]


        g_returns = self.adj_close[di-self.delay, valid_idx]/self.adj_close[di-self.delay-126, valid_idx] - 1
        g_close = self.adj_close[di-self.delay, valid_idx]
        group1 = OP.group_split2(g_returns, 5)
        group2 = OP.group_split2(g_close, 10)

        #returns1 = self.close[di-self.delay-63+1:di-self.delay+1, valid_idx] / self.close[di-self.delay-63:di-self.delay, valid_idx] - 1
        #returns1 = returns1 - np.nanmean(returns1)
        close1 = self.adj_close[di-self.delay-63:di-self.delay, valid_idx]/(self.start_price[valid_idx]+1e-5)
        close1 = close1 - np.nanmean(close1, axis=1, keepdims=True)
        std_returns1 = np.nanmax(close1, axis=0)/np.nanmin(close1, axis=0)-1#OP.std(returns1)
        tmp = np.zeros((1, self.adj_close[di].shape[0]))
        tmp[0, valid_idx] = std_returns1
        if(self.std_returns1 is None):
            self.std_returns1 = tmp.copy()
        elif(self.std_returns1.shape[0] < 21):
            self.std_returns1 = np.concatenate([self.std_returns1, tmp], axis=0)
        else:
            self.std_returns1 = np.concatenate([self.std_returns1[1:], tmp], axis=0)

        #returns2 = self.close[di-self.delay-21+1:di-self.delay+1, valid_idx] / self.close[di-self.delay-21:di-self.delay, valid_idx] - 1
        #returns2 = returns2 - np.nanmean(returns2)
        close2 = self.adj_close[di-self.delay-5:di-self.delay, valid_idx]/(self.start_price[valid_idx]+1e-5)
        close2 = close2 - np.nanmean(close2, axis=1, keepdims=True)
        std_returns2 = np.nanmax(close2, axis=0)/np.nanmin(close2, axis=0)-1
        #std_returns2 = OP.std(returns2)
        tmp = np.zeros((1, self.adj_close[di].shape[0]))
        tmp[0, valid_idx] = std_returns2
        if(self.std_returns2 is None):
            self.std_returns2 = tmp.copy()
        elif(self.std_returns2.shape[0] < 21):
            self.std_returns2 = np.concatenate([self.std_returns2, tmp], axis=0)
        else:
            self.std_returns2 = np.concatenate([self.std_returns2[1:], tmp], axis=0)

        m = self.std_returns1.shape[0]
        returns = self.adj_close[di-self.delay-m+1:di-self.delay+1, valid_idx] / self.adj_close[di-self.delay-m+1-21:di-self.delay+1-21, valid_idx] - 1
        #price = self.close[di-self.delay-m+1:di-self.delay+1, valid_idx]
        corr = OP.corr(self.std_returns1[:, valid_idx], self.std_returns2[:, valid_idx])
        #alpha = np.where((alpha > 1) & (alpha < 8), alpha, np.nan)
        alpha = -corr*OP.moment(returns, 4)
        # alpha = OP.rank(alpha) - 0.5
        alpha = OP.power(alpha, 2)
        # alpha = OP.groupNeutralize(alpha, group2)

        if(self.pre_alpha is None):
            self.alpha[valid_idx] = alpha
            self.pre_alpha = self.alpha.copy()
        else:
            mask1 = ~np.isnan(self.pre_alpha[valid_idx])
            mask2 = ~np.isnan(self.pre_alpha)
            if((di - self.delay)%1 == 0):
                self.alpha[valid_idx&mask2] = 0.2*self.pre_alpha[valid_idx&mask2] + 0.8*alpha[mask1]
                self.alpha[valid_idx&~mask2] = alpha[~mask1]
                self.pre_alpha = self.alpha.copy()
            else:
                self.alpha[valid_idx&mask2] = 1*self.pre_alpha[valid_idx&mask2] + 0*alpha[mask1]
                self.alpha[valid_idx&~mask2] = alpha[~mask1]
                self.pre_alpha = self.alpha.copy()

    def checkpointSave(self, fh):
        pickle.dump(self.pre_alpha, fh)
        pickle.dump(self.std_returns1, fh)
        pickle.dump(self.std_returns2, fh)
        pickle.dump(self.start_price, fh)

    def checkpointLoad(self, fh):
        self.pre_alpha = pickle.load(fh)
        self.std_returns1 = pickle.load(fh)
        self.std_returns2 = pickle.load(fh)
        self.start_price = pickle.load(fh)