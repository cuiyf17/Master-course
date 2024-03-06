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
        self.open = dr.getData('open')
        self.close = dr.getData('close')
        self.high = dr.getData('high')
        self.low = dr.getData('low')
        self.vwap = dr.getData('vwap')
        self.adj_volume = dr.getData('volume')
        self.adj_close = dr.getData('adj_close')
        self.adj_volume = dr.getData('adj_volume')
        self.adj_vwap = dr.getData('adj_vwap')
        self.adj_high = dr.getData('adj_high')
        self.adj_low = dr.getData('adj_low')
        self.adj_open = dr.getData('adj_open')
        self.amount = dr.getData('amount')
        self.industry = dr.getData('industry')
        self.family = dr.getData('family')
        self.zz500 = dr.getData('index.zz500')

        self.ihigh = dr.getData('interval5m.high')
        self.ilow = dr.getData('interval5m.low')	
        self.iclose = dr.getData('interval5m.close')
        self.iopen = dr.getData('interval5m.open')

        self.risk_free_return = 0
        self.pre_alpha = None #不可删除，用于动量更新控制换手率
        self.pre_alphas = None #不可删除，用于止盈止损
        self.pre_alphas_zscore = None #不可删除，用于zscore控制开仓平仓


    def generate(self, di):
        start_di = di - self.delay - self.lag - self.ndays + 1
        end_di = di - self.delay - self.lag + 1

        valid_idx = self.valid[di] #& self.valid[di-1] & (self.volume[di - 1] > 0) & (self.volume[di] > 0)
        # 停牌后复牌的股票不要
        valid_idx = valid_idx & ~((self.status[di-self.delay - 1] == 0) & (self.status[di-self.delay] == 1))
        # 前一天涨跌停的股票不要
        # mask1 = (np.abs(self.close[di-self.delay]/self.close[di-self.delay-1]-1) > 0.198) & (self.family[di-self.delay]==2)
        # mask2 = (np.abs(self.close[di-self.delay]/self.close[di-self.delay-1]-1) > 0.099) & (self.family[di-self.delay]!=2)
        # valid_idx = valid_idx & ~mask1 & ~mask2

        # 计算因子值
        g_returns = self.close[di-self.delay]/self.close[di-self.delay-21]-1
        group1 = OP.group_split2(g_returns, 10)

        ihigh = self.ihigh[start_di:end_di]
        ilow = self.ilow[start_di:end_di]
        iclose = self.iclose[start_di:end_di]
        iopen = self.iopen[start_di:end_di]
        iret = iclose/iopen-1
        maskwhere = ihigh/ilow - 1 < 0.02
        iret[maskwhere] = np.nan
        std = np.nanstd(iret, axis=1)

        # alpha = (ret[-1] - OP.mean(ret))/OP.std(ret)
        std = std/np.nanmean(std, axis=1, keepdims=True) - 1
        alpha = -OP.mean(std)*OP.std(std)
        # alpha = OP.rank(alpha) - 0.5
        # alpha = OP.power(alpha, 3)
        # alpha = np.where(np.abs(alpha) > 0.45, np.sign(alpha), np.nan)/alpha.shape[0]*10

        # 根据zscore开仓平仓
        alpha = self.zscore_position(alpha, di, openthres = 0, holdthres = 0., use_tail = True)
        # alpha = alpha*2
        # 止盈止损
        # alpha = self.corr_stop_profit_loss(alpha, di, lookback = 5, stop_profit = 0.8, stop_loss = -0.8, excess_zz500 = True, continuous = False)
        # 动量更新
        alpha = alpha[valid_idx]
        self.momentum_update(alpha, di, valid_idx, trade_interval = 1, momentum = 1)

    def checkpointSave(self, fh):
        pickle.dump(self.pre_alpha, fh)
        pickle.dump(self.pre_alphas, fh)
        pickle.dump(self.pre_alphas_zscore, fh)

    def checkpointLoad(self, fh):
        self.pre_alpha = pickle.load(fh)
        self.pre_alphas = pickle.load(fh)
        self.pre_alphas_zscore = pickle.load(fh)

    def zscore_position(self, alpha, di, openthres = 0.6, holdthres = 0.4, use_tail = False):
        tmp = np.zeros((1, self.close[di-1].shape[0]))
        tmp[0] = alpha
        if(self.pre_alphas_zscore is None):
            self.pre_alphas_zscore = tmp
        elif(self.pre_alphas_zscore.shape[0] < 256):
            self.pre_alphas_zscore = np.concatenate((self.pre_alphas_zscore, tmp), axis=0)
        else:
            self.pre_alphas_zscore = np.concatenate((self.pre_alphas_zscore[1:], tmp), axis=0)
        zscore = (alpha - np.nanmean(self.pre_alphas_zscore[:], axis=0))/np.nanstd(self.pre_alphas_zscore[:], axis=0)
        zscore = np.clip(zscore, -3, 3)/3
        if(self.pre_alpha is not None):
            choose_where = ((np.abs(zscore) > openthres) & (np.isnan(self.pre_alpha))) | ((np.abs(zscore) > holdthres) & (~np.isnan(self.pre_alpha)))
        else:
            choose_where = np.abs(zscore) > openthres
        if(use_tail):
            ones = np.where(choose_where, 1, np.nan)
            zscore[choose_where] = zscore[choose_where]/np.nansum(np.abs(zscore[choose_where]))*np.nansum(ones)
            alpha = np.where(choose_where, zscore, np.nan)/alpha.shape[0]
        else:
            alpha = np.where(choose_where, np.sign(zscore), np.nan)/alpha.shape[0]
        
        return alpha
    
    def corr_stop_profit_loss(self, alpha, di, lookback = 5, stop_profit = 0.7, stop_loss = -0.7, excess_zz500 = True, continuous = True):
        tmp = np.zeros((1, self.close[di-1].shape[0]))
        tmp[0] = alpha
        if(self.pre_alphas is None):
            self.pre_alphas = tmp
        elif(self.pre_alphas.shape[0] < lookback):
            self.pre_alphas = np.concatenate((self.pre_alphas, tmp), axis=0)
        else:
            self.pre_alphas = np.concatenate((self.pre_alphas[1:], tmp), axis=0)
        nn = self.pre_alphas.shape[0]
        if(excess_zz500):
            zz500_ret = (self.zz500[di-self.delay-nn+1:di-self.delay+1]/self.zz500[di-self.delay-nn:di-self.delay] - 1).reshape((-1, 1))
            ex_ret = self.close[di-self.delay-nn+1:di-self.delay+1]/self.close[di-self.delay-nn:di-self.delay] - 1 - zz500_ret
        else:
            ex_ret = self.close[di-self.delay-nn+1:di-self.delay+1]/self.close[di-self.delay-nn:di-self.delay] - 1
        alpha_momentum = OP.corr(ex_ret, self.pre_alphas[:])
        if(continuous):
            coef = np.zeros_like(alpha)+np.nan
            positive = np.where(alpha_momentum >= 0)
            negative = np.where(alpha_momentum < 0)
            coef[positive] = 1 - np.minimum(alpha_momentum[positive]/stop_profit, 1)
            coef[negative] = 1 - np.minimum(np.abs(alpha_momentum[negative]/stop_loss), 1)
            alpha = alpha * coef
        else:
            alpha = np.where((alpha_momentum > stop_loss) & (alpha_momentum < stop_profit), alpha, np.nan)
        
        return alpha
    
    def momentum_update(self, alpha, di, valid_idx, trade_interval = 1, momentum = 0.05):
        if(self.pre_alpha is None):
            self.alpha[valid_idx] = alpha
            self.pre_alpha = self.alpha.copy()
        else:
            mask1 = ~np.isnan(self.pre_alpha[valid_idx])
            mask2 = ~np.isnan(self.pre_alpha)
            if((di - self.delay)%trade_interval == 0):
                self.alpha[valid_idx&mask2] = (1-momentum)*self.pre_alpha[valid_idx&mask2] + momentum*alpha[mask1]
                self.alpha[valid_idx&~mask2] = alpha[~mask1]
                self.pre_alpha = self.alpha.copy()
            else:
                self.alpha[valid_idx&mask2] = 1*self.pre_alpha[valid_idx&mask2] + 0*alpha[mask1]
                self.alpha[valid_idx&~mask2] = alpha[~mask1]
                self.pre_alpha = self.alpha.copy()

