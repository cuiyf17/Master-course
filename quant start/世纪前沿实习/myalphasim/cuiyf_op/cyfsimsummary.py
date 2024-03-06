#!/usr/bin/env python3
import os, sys, re, math, argparse

RFRET = 0.15
TRADEDAY = 242

def doStats(XX, date, pnl, _long, short, ret, sh_hld, sh_trd, b_share, t_share):
    if (XX not in stats):
        stats[XX] = {}
        stats[XX]['dates'] = []
        stats[XX]['pnl'] = 0
        stats[XX]['long'] = 0
        stats[XX]['short'] = 0
        stats[XX]['sh_hld'] = 0
        stats[XX]['sh_trd'] = 0
        stats[XX]['avg_ret'] = 0
        stats[XX]['b_sh'] = 0
        stats[XX]['t_sh'] = 0
        stats[XX]['drawdown'] = 0
        stats[XX]['dd_start'] = '-1'
        stats[XX]['dd_end'] = '-1'
        stats[XX]['up_days'] = 0
        stats[XX]['days'] = 0
        stats[XX]['xsy'] = 0
        stats[XX]['xsyy'] = 0

    stats[XX]['dates'].append(date)
    stats[XX]['pnl'] += pnl
    stats[XX]['long'] += _long
    stats[XX]['short'] += short
    stats[XX]['sh_hld'] += sh_hld
    stats[XX]['sh_trd'] += sh_trd
    stats[XX]['avg_ret'] += (ret- RFRET/TRADEDAY)
    stats[XX]['b_sh'] += b_share
    stats[XX]['t_sh'] += t_share
    stats[XX]['xsy'] += (ret- RFRET/TRADEDAY)
    stats[XX]['xsyy'] += (ret- RFRET/TRADEDAY)**2
    if (pnl > 0):
        stats[XX]['up_days'] += 1
    if (_long != 0 or short != 0):
        stats[XX]['days'] += 1


# global variables
LONGSHORT_SCALE = 1e6
PNL_SCALE = 1e6

DD_start = '-1'
DD_setst = 1
DD_sum = 0
stats = {}

pnlfile = ''
type = 'yearly'
sdate = -1
edate = -1
pattern = re.compile(r'\s+')

parser = argparse.ArgumentParser(description='simsummary')
parser.add_argument('-s', '--start', type=int, help='start date', default=-1)
parser.add_argument('-e', '--end', type=int, help='end date', default=-1)
parser.add_argument('pnl', type=str, help='pnl file')
parser.add_argument('-t', '--type', choices=['yearly', 'monthly'], help='summary type', default='yearly')
parser.add_argument('-r', '--ret', type=float, default=0)
args = parser.parse_args()
sdate = args.start
edate = args.end
pnlfile = args.pnl
type = args.type
RFRET = args.ret
# try:
#     opts, args = getopt.getopt(sys.argv[1:], "s:e:p:t:r", ["sdate=", "edate=", "pnl=", "type=", "ret="])
#     for o, a in opts:
#         if o in ("-s", "--sdate"):
#             sdate = a
#         elif o in ("-e", "--edate"):
#             edate = a
#         elif o in ("-p", "--pnl"):
#             pnlfile = os.path.abspath(a)
#         elif o in ("-t", "--type"):
#             type = a
#         elif o in ("-r", "--ret"):
#             RFRET = a
#     if pnlfile == '':
#         pnlfile = os.path.abspath(args[0])
# except getopt.GetoptError as err:
#     print(err)
#     sys.exit(0)

if (os.path.isfile(pnlfile) == False):
    print("Pnl file '%s' does not exist" % (pnlfile))
    sys.exit(-1)

with open(pnlfile, 'r') as f:
    for line in f:
        line = line.strip('\n')
        if len(line) < 10:
            continue
        date, pnl, _long, short, ret, sh_hld, sh_trd, b_share, t_share, other = pattern.split(line, 9)
        date = date[:8]
        if (int(sdate) > 0 and int(date) < int(sdate)) or (int(edate) > 0 and int(date) > int(edate)):
            continue
        XX = date[0:4]
        if type == 'monthly':
            XX = date[0:6]
        doStats(XX, date, float(pnl), float(_long), float(short), float(ret), float(sh_hld), float(sh_trd),
                float(b_share), float(t_share))
        doStats('ALL', date, float(pnl), float(_long), float(short), float(ret), float(sh_hld), float(sh_trd),
                float(b_share), float(t_share))

        # drawdown
        if (DD_setst == 1):
            DD_start = date
            DD_setst = 0
        DD_sum += (float(pnl)-float(_long)*RFRET/TRADEDAY)
        if (DD_sum >= 0):
            DD_sum = 0
            DD_start = date
            DD_setst = 1

        if (DD_sum < stats[XX]['drawdown']):
            stats[XX]['drawdown'] = DD_sum
            stats[XX]['dd_start'] = DD_start
            stats[XX]['dd_end'] = date

        if (DD_sum < stats['ALL']['drawdown']):
            stats['ALL']['drawdown'] = DD_sum
            stats['ALL']['dd_start'] = DD_start
            stats['ALL']['dd_end'] = date

# print head line
print("%17s %7s %8s %7s %7s %7s %14s %5s %5s %7s %9s %9s" % ("dates", "long(M)", "short(M)", "pnl(M)", "%ret", "%tvr", "shrp (IR)", "%dd", "%win", 'fitness','ddStart','ddEnd'))

for XX in sorted(stats):
    if XX == 'ALL':
        print('')
    d = float(stats[XX]['days'])
    # print XX, d
    if (d < 1):
        continue
    _long = stats[XX]['long'] / d
    short = stats[XX]['short'] / d
    ret = stats[XX]['avg_ret'] / d * TRADEDAY
    perwin = stats[XX]['up_days'] / d
    turnover = stats[XX]['sh_trd'] / stats[XX]['sh_hld'] if (stats[XX]['sh_hld'] > 0) else 0
    # drawdown = stats[XX]['drawdown'] / _long * -100 if (_long > 0) else 0
    drawdown = stats[XX]['drawdown'] / stats[XX]['sh_hld']*d*2 * -100 if (_long > 0) else 0
    ir = 0
    if (d > 2):
        avg = stats[XX]['xsy'] / d
        std = math.sqrt((stats[XX]['xsyy'] - stats[XX]['xsy'] * stats[XX]['xsy'] / d) / (d - 1))
        ir = avg / std if (std > 0) else 0

    # fitness = ir / turnover if (turnover > 0) else 0
    fitness = ir * math.sqrt(TRADEDAY) * math.sqrt(math.fabs(ret / turnover)) if (turnover > 0) else 0
    margin = stats[XX]['pnl'] / stats[XX]['sh_trd'] * 10000 if (stats[XX]['sh_trd'] > 0) else 0
    margin_cs = stats[XX]['pnl'] / stats[XX]['t_sh'] * 100 if (stats[XX]['t_sh'] > 0) else 0

    #  print "%8d-%8d %7.2f %8.2f %7.3f %7.2f %7.2f %6.2f(%4.2f) %5.2f %5.2f %6.2f %6.2f %7.2f" % (stats[XX]['dates'][0], stats[XX]['dates'][-1], _long / LONGSHORT_SCALE, short / LONGSHORT_SCALE, stats[XX]['pnl'] / PNL_SCALE, ret * 100, turnover * 100, ir * math.sqrt(TRADEDAY), ir, drawdown, perwin, margin, margin_cs, fitness)
    print(
        f"{stats[XX]['dates'][0]:8s}-{stats[XX]['dates'][-1]:8s} {_long / LONGSHORT_SCALE:7.2f} {short / LONGSHORT_SCALE:8.2f} {stats[XX]['pnl'] / PNL_SCALE:7.3f} {ret * 100:7.2f} {turnover * 100:7.2f} {ir * math.sqrt(TRADEDAY):7.2f} ({ir:4.2f}) {drawdown:5.2f} {perwin:5.2f} {fitness:7.2f} {stats[XX]['dd_start']:>9s} {stats[XX]['dd_end']:>9s}")
