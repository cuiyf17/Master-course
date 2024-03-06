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
        self.status = dr.getData('status')
        self.cap = dr.getData('cap')
        self.negcap = dr.getData('negcap')
        self.close = dr.getData('adj_close')
        self.volume = dr.getData('adj_volume')
        self.vwap = dr.getData('adj_vwap')
        self.high = dr.getData('adj_high')
        self.low = dr.getData('adj_low')
        self.open = dr.getData('adj_open')
        self.amount = dr.getData('amount')
        self.subindustry = dr.getData('subindustry')
        self.industry = dr.getData('industry')
        self.sector = dr.getData('sector')
        self.family = dr.getData('family')
        self.country = dr.getData('country')
        self.risk_free_return = 0
        self.pre_alpha = None
        self.upshadow = None
        self.downshadow = None


    def generate(self, di):
        start_di = di - self.delay - self.lag - self.ndays + 1
        end_di = di - self.delay - self.lag + 1

        valid_idx = self.valid[di] #& self.valid[di-1] & (self.volume[di - 1] > 0) & (self.volume[di] > 0)
        #停牌后复牌的股票不要
        valid_idx = valid_idx & ~((self.status[di - 1] == 0) & (self.status[di] == 1))
        #前一天涨跌停的股票不要
        #mask1 = (np.abs(self.close[di-self.delay]/self.close[di-self.delay-1]-1) > 0.195) & (self.family[di-self.delay]==2)
        #mask2 = (np.abs(self.close[di-self.delay]/self.close[di-self.delay-1]-1) > 0.098) & (self.family[di-self.delay]!=2)
        #valid_idx = valid_idx & ~mask1 & ~mask2


        subindustry = self.subindustry[di-self.delay, valid_idx]
        g_returns = self.close[di-self.delay, valid_idx]/self.close[di-self.delay-21, valid_idx] - 1
        g_close = self.close[di-self.delay, valid_idx]
        g_cap = self.cap[di-self.delay, valid_idx]
        g_hot = self.amount[di-self.delay, valid_idx]
        group1 = OP.group_split2(g_returns, 10)
        group2 = OP.group_split2(g_close, 10)
        group3 = OP.group_split2(g_cap, 10)
        group4 = OP.group_split2(g_hot, 10)

        open = self.open[di-self.delay, valid_idx]
        close = self.close[di-self.delay, valid_idx]
        high = self.high[di-self.delay, valid_idx]
        low = self.low[di-self.delay, valid_idx]
        
        openclose = np.concatenate([open.reshape(-1, 1), close.reshape(-1, 1)], axis=1)
        upshadow = (high - np.nanmax(openclose, axis=1))/close
        upshadow = upshadow - OP.group_mean(upshadow, subindustry)
        downshadow = (np.nanmin(openclose, axis=1) - low)/close
        downshadow = downshadow - OP.group_mean(downshadow, subindustry)
        
        tmp1 = np.zeros((1, self.close[di-self.delay].shape[0])) + np.nan
        tmp1[0, valid_idx] = upshadow
        tmp2 = np.zeros((1, self.close[di-self.delay].shape[0])) + np.nan
        tmp2[0, valid_idx] = downshadow
        if(self.upshadow is None):
            self.upshadow = tmp1
            self.downshadow = tmp2
        elif(self.upshadow.shape[0] < self.ndays):
            self.upshadow = np.concatenate([self.upshadow, tmp1], axis=0)
            self.downshadow = np.concatenate([self.downshadow, tmp2], axis=0)
        else:
            self.upshadow = np.concatenate([self.upshadow[1:], tmp1], axis=0)
            self.downshadow = np.concatenate([self.downshadow[1:], tmp2], axis=0)

        alpha = -OP.corr(self.upshadow[:, valid_idx], self.downshadow[:, valid_idx])
        alpha = alpha - np.nanmean(alpha)
        # alpha = OP.rank(alpha) - 0.5
        alpha = OP.power(alpha, 4)

        # alpha = OP.group_mean(alpha, industry)
        alpha = OP.groupNeutralize(alpha, group2)
        alpha = OP.groupNeutralize(alpha, group3, standardization = False)


        if(self.pre_alpha is None):
            self.alpha[valid_idx] = alpha
            self.pre_alpha = self.alpha.copy()
        else:
            mask1 = ~np.isnan(self.pre_alpha[valid_idx])
            mask2 = ~np.isnan(self.pre_alpha)
            if((di - self.delay)%1 == 0):
                self.alpha[valid_idx&mask2] = 0.4*self.pre_alpha[valid_idx&mask2] + 0.6*alpha[mask1]
                self.alpha[valid_idx&~mask2] = alpha[~mask1]
                self.pre_alpha = self.alpha.copy()
            else:
                self.alpha[valid_idx&mask2] = 1*self.pre_alpha[valid_idx&mask2] + 0*alpha[mask1]
                self.alpha[valid_idx&~mask2] = alpha[~mask1]
                self.pre_alpha = self.alpha.copy()

    def checkpointSave(self, fh):
        pickle.dump(self.pre_alpha, fh)
        pickle.dump(self.upshadow, fh)
        pickle.dump(self.downshadow, fh)


    def checkpointLoad(self, fh):
        self.pre_alpha = pickle.load(fh)
        self.upshadow = pickle.load(fh)
        self.downshadow = pickle.load(fh)