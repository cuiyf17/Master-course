# %%
import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# %%
portfolio_path = os.path.abspath(os.path.join(os.getcwd(), "./附件1-portfolio.csv"))
quote_path = os.path.abspath(os.path.join(os.getcwd(), "./附件2-quote.csv"))

portfolio = pd.read_csv(portfolio_path)
quote = pd.read_csv(quote_path)
quote['date'] = pd.to_datetime(quote['date'])

# %%
start_date = "2021-11-01"
end_date = "2021-11-30"

# %%
weight_dict = dict()
for i in range(len(portfolio)):
    weight_dict[str(portfolio['code'][i])] = portfolio['weight'][i]

# %%
def cal_return(stock_dict, start_date, end_date):
    stock_price = stock_dict.copy()
    
    idx_start = stock_price[stock_price['date'] >= start_date].index[0]
    idx_end = stock_price[stock_price['date'] <= end_date].index[-1]
    stock_price["return"] = stock_price["close"].div(stock_price["close"].shift(1)) - 1
    ret =  stock_price[["date", "return"]].loc[idx_start:idx_end]
    #print(stock_price[["date", "return"]])
    #print(ret)
    return ret


# %%
stock_dict = dict()
return_dict = dict()
groups = quote.groupby("code")
for item in groups:
    stock_dict[item[0]] = item[1].reset_index(drop=True)
    return_dict[item[0]] = cal_return(stock_dict[item[0]], start_date, end_date)

# %%
portfolio_return = None
for code in weight_dict:
    if portfolio_return is None:
        portfolio_return = return_dict[code].copy()
    else:
        portfolio_return['return'] += return_dict[code]['return'] * weight_dict[code]
portfolio_return["excessReturn"] = portfolio_return["return"] - return_dict["sh000001"]["return"]


# %%
excess_return = portfolio_return[["date", "excessReturn"]].round(4)
#excess_return["excessReturn"] = excess_return["excessReturn"].round(4)

# %%
excess_return.to_csv("A2+崔晏菲.csv", index=False)