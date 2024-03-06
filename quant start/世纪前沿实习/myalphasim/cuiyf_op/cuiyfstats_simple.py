import alphasim.data.Universe as uv
from alphasim.data import DataRegistry as dr
from alphasim.stats import StatsBase
from alphasim.utils import Oputil
import numpy as np
import os
from alphasim import Checkpoint as cpt
import pickle

import cuiyf_op.cuiyfOp as OP

'''
Simple stats for long x short

limit hitting & halt: keep positions

20180409: remove adj dependency & support D0. DO NOT use vwap=true for D0.
20200301: change cost calculation
20210227: change price limit calculation, and apply price limit/suspend by default.
'''


class cuiyfStatsSimple(StatsBase):
    def __init__(self, cfg, alphaNode):
        StatsBase.__init__(self, alphaNode)
        self.printStats = cfg.getAttributeDefault('printStats', True)
        self.dumpPnl = cfg.getAttributeDefault('dumpPnl', True)
        self.maxjump = cfg.getAttributeDefault('maxjump', 4.0)
        self.anndays = cfg.getAttributeDefault('anndays', 242)  # num of trading days: ~244? need to calculate from data
        self.tax = cfg.getAttributeDefault('tax', 0.)
        self.fee = cfg.getAttributeDefault('fee', 0.)
        self.slippage = cfg.getAttributeDefault('slippage', 0.)
        self.keepHalt = cfg.getAttributeDefault('keepHalt', True)
        self.maxHaltDays = cfg.getAttributeDefault('maxHaltDays', 250)
        self.costSingle = cfg.getAttributeDefault('costSingle', False)
        # if self.dumpPnl:
        #     pnlDir = cfg.getAttributeStringDefault('pnlDir', './pnl')
        #     if not os.path.isdir(pnlDir):
        #         os.makedirs(pnlDir)
        #     pnlName = self.alphaNode.alphaId
        #     pnlFile = pnlDir + '/' + pnlName
        #     self.output = open(pnlFile, 'r+' if os.path.isfile(pnlFile) and cpt.cptdir!='' else 'w', buffering=4096)
        if len(self.alphaNode.children) == 0:
            if not self.costSingle:  # do not apply cost to single alpha
                self.tax = 0.
                self.fee = 0.
                self.slippage = 0.
        # load data to calculate stats: cps, rawcps, returns, vwap, currency, etc.
        self.cps = dr.getData('close')
        self.adj = dr.getData('adjfactor')
        self.ops = dr.getData('open')
        self.vwap = dr.getData('vwap')
        self.volume = dr.getData('volume')
        self.uplmt = dr.getData('upper_limit')
        self.lolmt = dr.getData('lower_limit')
        self.tpname = cfg.getAttributeStringDefault('tradePrice', 'close')
        self.trdp = dr.getData(self.tpname)
        # variables for stats calculation
        self.oPos = np.full(len(uv.Instruments), np.nan)
        self.cumPnl = 0.
        self.cumHldVal = 0.
        self.cumTrdVal = 0.
        self.ddSum = 0.
        self.ddMax = 0.
        self.yestlong = 0.
        self.yestshrt = 0.
        self.num = 0
        self.mn = 0.
        self.var = 0.  # sample variance = sum_i_n(xi - mean(x))^2 / (n - 1). For computing simplicity, 1 / (n - 1) is dropped here.
        self.cumTrdPnl = 0.  # trading pnl

        # stats for combo
        self.ret = 0
        self.avg_ret = 0
        self.ir = np.nan
        self.tvr = np.nan
        self.ddPct = 0.
        self.ddMaxPct = 0.

        self.long_value = 0.
        self.shrt_value = 0.
        self.long_cnt = 0
        self.shrt_cnt = 0
        self.trd_val = 0.
        self.hld_val = 0.
        self.trd_shr = 0.
        self.hld_shr = 0.
        self.pnl = 0.

        self.alphaname = self.alphaNode.alphaId
        now_path = os.getcwd()
        if not os.path.exists(now_path + '/alpha_values'):
            os.makedirs(now_path + '/alpha_values')
        self.alpha_path = now_path + '/alpha_values/' + self.alphaname
        with open(self.alpha_path, 'w') as f:
            f.write('')

        self.zz500 = dr.getData('index.zz500')
        if self.dumpPnl:
            pnlDir = cfg.getAttributeStringDefault('pnlDir', 'pnl')
            if not os.path.isdir(now_path + "/" + pnlDir):
                os.makedirs(now_path + "/" + pnlDir)
            self.pnlFile = now_path + "/" + pnlDir + '/' + self.alphaname
            with open(self.pnlFile, 'w') as f:
                f.write('')
        self.dump_group_pnl = cfg.getAttributeDefault("dump_group_pnl", False)
        if(self.dump_group_pnl):
            group_retDir = cfg.getAttributeStringDefault('group_retDir', 'group_ret')
            if not os.path.isdir(now_path + "/" + group_retDir):
                os.makedirs(now_path + "/" + group_retDir)
            self.group_retFile = now_path + "/" + group_retDir + '/' + self.alphaname
            with open(self.group_retFile, 'w') as f:
                f.write('')

            group_pnlDir = cfg.getAttributeStringDefault('group_pnlDir', 'group_pnl')
            if not os.path.isdir(now_path + "/" + group_pnlDir):
                os.makedirs(now_path + "/" + group_pnlDir)
            self.group_pnlFile = now_path + "/" + group_pnlDir + '/' + self.alphaname
            with open(self.group_pnlFile, 'w') as f:
                f.write('')

            longshort_pnlDir = cfg.getAttributeStringDefault('longshort_pnlDir', 'longshort_pnl')
            if not os.path.isdir(now_path + "/" + longshort_pnlDir):
                os.makedirs(now_path + "/" + longshort_pnlDir)
            self.longshort_pnlFile = now_path + "/" + longshort_pnlDir + '/' + self.alphaname
            with open(self.longshort_pnlFile, 'w') as f:
                f.write('')
        self.num_groups = int(cfg.getAttributeDefault("num_groups", 10))

    def calculate(self, di):
        with open(self.alpha_path, 'a') as f:
            alphavalue = self.alphaNode.alpha[~np.isnan(self.alphaNode.alpha)]
            line = " ".join([str(i) for i in alphavalue])
            f.write(line + '\n')

        preclose = self.cps[di - 1] * self.adj[di]
        nPos = np.copy(self.alphaNode.alpha)  # stats is not supposed to change alpha vector
        # limit hitting & halt
        book_p = np.nansum(nPos[nPos >= 0])
        book_n = np.fabs(np.nansum(nPos[nPos < 0]))
        idx_lmt = (self.ops[di] >= self.uplmt[di] - 0.01) | (self.ops[di] <= self.lolmt[di] + 0.01)
        nPos[idx_lmt] = np.nan
        freeze_p = np.nansum(self.oPos[idx_lmt & (self.oPos >= 0)])
        freeze_n = np.fabs(np.nansum(self.oPos[idx_lmt & (self.oPos < 0)]))
        idx_hlt = ~(self.volume[di] > 0)
        if self.keepHalt:
            idx_hlt &= (np.sum(self.volume[di - self.maxHaltDays + 1:di + 1] > 0, axis=0) > 0)  # exclude zombies
            freeze_p += np.nansum(self.oPos[idx_hlt & (self.oPos >= 0)])
            freeze_n += np.fabs(np.nansum(self.oPos[idx_hlt & (self.oPos < 0)]))
        nPos[idx_hlt] = np.nan  # do not buy halted stocks in any case
        if book_p < freeze_p:
            book_n = book_p + book_n - freeze_p - freeze_n
            book_p = 0
        elif book_n < freeze_n:
            book_p = book_p + book_n - freeze_p - freeze_n
            book_n = 0
        else:
            book_p -= freeze_p
            book_n -= freeze_n
        idx_pos = nPos >= 0
        pos_p = nPos[idx_pos]
        pos_n = nPos[~idx_pos]
        Oputil.scaleToBook(pos_p, book_p)
        Oputil.scaleToBook(pos_n, book_n)
        nPos[idx_pos] = pos_p
        nPos[~idx_pos] = pos_n
        nPos[idx_lmt] = self.oPos[idx_lmt]
        if self.keepHalt:
            nPos[idx_hlt] = self.oPos[idx_hlt]
        # diff position
        _ret = self.cps[di] / preclose - 1
        _ret[~(self.volume[di] > 0)] = np.nan
        extreme = (_ret > self.maxjump) | (_ret < -1 * self.maxjump / (self.maxjump + 1))
        nPos[extreme] = np.nan  # to replicate XXSim's result for extreme events
        self.oPos[extreme] = np.nan
        today_dollar_weight = np.full(nPos.size, 0.)
        today_dollar_weight[~np.isnan(nPos)] = nPos[~np.isnan(nPos)]
        yesterday_dollar_weight = np.full(nPos.size, 0.)
        yesterday_dollar_weight[~np.isnan(self.oPos)] = self.oPos[~np.isnan(self.oPos)]
        trades = today_dollar_weight - yesterday_dollar_weight
        trades[~(self.volume[di] > 0)] = np.nan  # no pnl for halted stocks; little effects
        trade_price = np.copy(self.trdp[di])
        trade_price[trade_price < 1e-6] = np.nan  # exclude 0, 20160504
        trade_price[trades > 0] *= (1.0 + self.slippage / 10000.)
        trade_price[trades < 0] *= (1.0 - self.slippage / 10000.)
        # value, # of stock and shares
        self.long_value = np.nansum(today_dollar_weight[today_dollar_weight > 0])
        self.long_cnt = np.sum(today_dollar_weight > 0)
        self.shrt_value = np.nansum(today_dollar_weight[today_dollar_weight < 0])
        self.shrt_cnt = np.sum(today_dollar_weight < 0)
        val_t = np.fabs(trades)
        val_h = np.fabs(yesterday_dollar_weight)
        self.trd_val = np.nansum(val_t)
        self.hld_val = np.nansum(val_h)
        self.trd_shr = np.nansum(val_t / preclose)
        self.hld_shr = np.nansum(val_h / preclose)
        # pnl
        if(self.dump_group_pnl):
            zz500_ret = self.zz500[di] / self.zz500[di-1] - 1
            longshort_pnl = np.zeros(4)
            for i in range(2):
                idx = yesterday_dollar_weight >= 0 if i == 0 else yesterday_dollar_weight < 0
                longshort_pnl[i] += np.nansum((-1)**i*yesterday_dollar_weight[idx]*_ret[idx])
                longshort_pnl[i+2] += np.nansum((-1)**i*yesterday_dollar_weight[idx]*zz500_ret)
        hld_pnl = np.nansum(self.oPos * _ret)
        trd_pnl = np.nansum(trades * (self.cps[di] / trade_price - 1))
        cost = np.nansum(np.fabs(trades *np.where(trades > 0, self.fee / 10000., (self.fee + self.tax) / 10000.)))
        trd_pnl -= cost
        self.pnl = hld_pnl + trd_pnl
        self.cumPnl += self.pnl
        self.cumTrdPnl += trd_pnl
        self.oPos[:] = nPos[:]  # update position
        # turnover
        self.cumHldVal += self.yestlong - self.yestshrt
        self.cumTrdVal += self.trd_val
        self.tvr = np.inf
        if self.num > 0 and self.cumHldVal > 0: self.tvr = self.cumTrdVal / self.cumHldVal
        # drawdown
        self.ddSum += self.pnl
        if self.ddSum > 0: self.ddSum = 0
        if self.ddSum < self.ddMax: self.ddMax = self.ddSum
        self.ddPct = np.nan
        if self.num > 0 and self.cumHldVal > 0:
            self.ddPct = self.ddSum / (self.cumHldVal / self.num / 2.)  # percent of long side
        self.ddMaxPct = np.nan
        if self.num > 0 and self.cumHldVal > 0:
            self.ddMaxPct = self.ddMax / (self.cumHldVal / self.num / 2.)  # percent of long side
        # ret & ir
        self.ret = 0.
        if self.num > 0 and self.yestlong - self.yestshrt > 0:
            self.ret = 2 * self.pnl / (self.yestlong - self.yestshrt)
        if not np.isnan(self.ret):
            self.num += 1
            ratio = (self.num - 1) / self.num
            self.var = self.var + self.mn * self.mn * ratio - 2 * self.ret * self.mn * ratio + self.ret * self.ret * ratio
            self.mn = self.mn * ratio + self.ret / self.num
        self.avg_ret = self.mn * self.anndays
        self.ir = np.nan
        if self.num > 1: self.ir = self.mn / np.sqrt(self.var / (self.num - 1))

        self.yestlong = self.long_value
        self.yestshrt = self.shrt_value

        # screen output
        if self.printStats:
            print('%8d%32s%12d x %-10d %-5s x %5s %10d %10d %12d %7.3f %7.3f %7.3f %10.3f' % (
            uv.Dates[di], self.alphaNode.alphaId, int(self.long_value),
            int(self.shrt_value), self.long_cnt, self.shrt_cnt,
            int(self.trd_shr), int(self.pnl), int(self.cumPnl), self.tvr,
            self.avg_ret, self.ddMaxPct, self.ir))
        if self.tpname != 'close' or self.fee > 0 or self.tax > 0:
            print('cumulative trade pnl: %.1f' % self.cumTrdPnl)

        # pnl file
        # format: date pnl long short return heldvalue tradevalue heldshares tradedshares longcount shortcount
        # if self.dumpPnl:
        #     pnlLine = '%d %f %f %f %f %f %f %f %f %d %d\n' % (
        #     uv.Dates[di], self.pnl, self.long_value, self.shrt_value, self.ret, self.hld_val, self.trd_val,
        #     self.hld_shr, self.trd_shr, self.long_cnt, self.shrt_cnt)
        #     self.output.write(pnlLine)
        if self.dumpPnl:
            pnlLine = '%d %f %f %f %f %f %f %f %f %d %d\n' % (
            uv.Dates[di], self.pnl, self.long_value, self.shrt_value, self.ret, self.hld_val, self.trd_val,
            self.hld_shr, self.trd_shr, self.long_cnt, self.shrt_cnt)
            with open(self.pnlFile, 'a') as f:
                f.write(pnlLine)
        if self.dump_group_pnl:
            longshort_pnlLine = '%d %s\n'%(uv.Dates[di], ' '.join([str(x) for x in longshort_pnl]))
            with open(self.longshort_pnlFile, 'a') as f:
                f.write(longshort_pnlLine)
        return

    def get(self, name):
        if name == 'long_value': return self.long_value
        if name == 'shrt_value': return self.shrt_value
        if name == 'pnl':        return self.pnl
        if name == 'cum_pnl':    return self.cumPnl
        if name == 'tvr':        return self.tvr
        if name == 'ret':        return self.ret
        if name == 'avg_ret':    return self.avg_ret
        if name == 'ddPct':      return self.ddPct * 100
        if name == 'ddMaxPct':   return self.ddMaxPct * 100
        if name == 'ir':         return self.ir
        if name == 'hld_val':    return self.hld_val
        if name == 'trd_val':    return self.trd_val
        if name == 'hld_shr':    return self.hld_shr
        if name == 'trd_shr':    return self.trd_shr
        if name == 'long_cnt':  return self.long_cnt
        if name == 'shrt_cnt':  return self.shrt_cnt
        return np.nan

    def checkpointSave(self, fh):
        pickle.dump(self.oPos, fh)
        pickle.dump(self.cumPnl, fh)
        pickle.dump(self.cumHldVal, fh)
        pickle.dump(self.cumTrdVal, fh)
        pickle.dump(self.ddSum, fh)
        pickle.dump(self.ddMax, fh)
        pickle.dump(self.yestlong, fh)
        pickle.dump(self.yestshrt, fh)
        pickle.dump(self.num, fh)
        pickle.dump(self.mn, fh)
        pickle.dump(self.var, fh)
        return

    def checkpointLoad(self, fh):
        self.oPos = pickle.load(fh)
        self.cumPnl = pickle.load(fh)
        self.cumHldVal = pickle.load(fh)
        self.cumTrdVal = pickle.load(fh)
        self.ddSum = pickle.load(fh)
        self.ddMax = pickle.load(fh)
        self.yestlong = pickle.load(fh)
        self.yestshrt = pickle.load(fh)
        self.num = pickle.load(fh)
        self.mn = pickle.load(fh)
        self.var = pickle.load(fh)

        # old pnl
        # if(self.dumpPnl):
        #     line = self.output.readline()
        #     while line:
        #         dt = int(line.split()[0])
        #         if dt == uv.Dates[cpt.savedi]:
        #             self.output.seek(self.output.tell())
        #             break
        #         else:
        #             line = self.output.readline()
        
        return
