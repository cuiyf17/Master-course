import os
import sys
import pdb
import csv
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle
from alphasim import AlphaOpBase
from alphasim import Oputil
from alphasim import DataRegistry as dr
from alphasim.data import Universe as uv

# 将其他路径添加到sys.path中
sys.path.append(os.path.abspath('/home/cuiyf/myalphasim/'))
import cuiyf_op.cuiyfOp as OP

def DECAYLINEAR(A):
    inds = A.shape[0]
    cols = A.shape[1]
    w=np.arange(1,inds+1).repeat(cols).reshape((inds,cols))
    w = np.where(~np.isnan(A),w,np.nan)
    w=w/np.nansum(w,axis=0)
    return np.nansum(A*w,axis=0)



class cuiyfSaveGroupProfit(AlphaOpBase):
    # 这个模块并没有计算交易费用
    def __init__(self, cfg):
        AlphaOpBase.__init__(self, cfg)
        self.tradeprice = cfg.getAttributeStringDefault('tradeprice', 'close')
        self.groupname = cfg.getAttributeStringDefault('group', 'alpha')
        self.num_group = cfg.getAttributeDefault('num_group', 10)
        self.available_groups = {'alpha', 'country', 'family', 'sector', 'industry', 'subindustry'}
        if(self.groupname not in self.available_groups):
            raise ValueError('group must be one of {}'.format(self.available_groups))
        elif(self.groupname == 'alpha'):
            self.group = np.zeros(self.num_group)
        else:
            self.group = dr.getData(self.group)
            self.num_group = np.unique(self.group).shape[0]
        self.booksize = float(cfg.getConfig().getparent().getparent().attrib['booksize'])
        self.savepath = cfg.getAttributeStringDefault('savepath', '/home/cuiyf/myalphasim/groupprofit.csv')
        if(os.path.exists(self.savepath)):
            os.remove(self.savepath)
            with open(self.savepath, 'w') as f:
                pass
        self.price = dr.getData(self.tradeprice)
        self.close = dr.getData('close')
        
        self.last_alpha = None
        self.cum_pnl = 0
        self.totalbook = None
        #self.iclose = dr.getData('interval5m.close')
        #self.iopen = dr.getData('interval5m.open')
        #self.iamount = dr.getData('interval5m.amo')
        #self.industry = dr.getData('industry')
        #self.SS2 = dr.getData('ZZ500')

    def apply(self, di, alpha):
        if(self.totalbook is None):
            self.totalbook = self.booksize
        else:
            self.totalbook = np.nansum(self.last_alpha/self.price[di-1]*self.price[di])
        alpha_min = np.nanmin(alpha)
        alpha_max = np.nanmax(alpha)
        alpha_sum = np.nansum(alpha[np.where(alpha>=0)]) - np.nansum(alpha[np.where(alpha<0)])
        tmp_alpha = alpha/alpha_sum * self.booksize
        group_profit = np.zeros(self.num_group)

        if(self.last_alpha is None):
            self.last_alpha = tmp_alpha.copy()
        else:
            last_stock = self.last_alpha/self.price[di-1]
            this_price = last_stock*self.close[di]
            profit = this_price - self.last_alpha
            
            if(self.groupname == 'alpha'):
                group = OP.group_split2(self.last_alpha, self.num_group)
                for i in range(self.num_group):
                    group_profit[i] = np.nansum(profit[np.where(group == i)])
            else:
                for i in range(self.num_group):
                    group_profit[i] = np.nansum(profit[np.where(self.group[di-1] == i)])

            self.last_alpha = tmp_alpha.copy()

        print("OP cal pnl = %f, cumpnl = %f, long = %f, short = %f"%(np.nansum(group_profit), self.cum_pnl, np.nansum(tmp_alpha[np.where(tmp_alpha>=0)]), np.nansum(tmp_alpha[np.where(tmp_alpha<0)])))
        with open(self.savepath, 'a', newline='') as f:
            writer = csv.writer(f)
            # 写入数据
            writer.writerow(group_profit)



    def checkpointSave(self, fh):
        pickle.dump(self.last_alpha, fh)
        pickle.dump(self.cum_pnl, fh)
        pickle.dump(self.totalbook, fh)

    def checkpointLoad(self, fh):
        self.last_alpha = pickle.load(fh)
        self.cum_pnl = pickle.load(fh)
        self.totalbook = pickle.load(fh)
