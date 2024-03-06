import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def simsummary(pnl_path):
    os.system('simsummary ' + pnl_path)

def bcorr(pnl_path, target_folder):
    if(type(target_folder) == str):
        target_folder = [target_folder]
    
    roots = [os.path.abspath(x) for x in target_folder]
    for root in roots:
        print(root + ":")
        target_pnls = [x for x in os.listdir(root) if not (x.endswith('.xml') or x.endswith('.csv') or x.endswith('.txt') or x.endswith('.pkl') or x.endswith('.ipynb') or x.endswith('.py') or x.endswith('.checkpoint'))]
        max_len = [len(x) for x in target_pnls]
        max_len = max(max_len)
        for name in target_pnls:
            target_pnl = os.path.join(root, name)
            tmp_str = "%-" + str(max_len) + "s"
            print("    " + tmp_str%(name), end=": ")
            os.system("python /home/cuiyf/myalphasim/cuiyf_op/bcorr.py " + pnl_path + " " + target_pnl + " 2 2")

def visualize_pnl(pnl_path):
    alpha_name = os.path.basename(pnl_path)
    columns = ["dates", "pnl", "long", "short", "sh_hold", "sh_trd", "avg_ret", "b_sh", "t_sh", "xsy", "xsyy"]
    df_pnl = pd.read_csv(pnl_path, sep=" ", header=None)
    df_pnl.columns = columns
    df_pnl["dates"] = pd.to_datetime(df_pnl["dates"], format="%Y%m%d")
    df_pnl["cumpnl"] = df_pnl["pnl"].cumsum()
    '''
    from ipywidgets import Output, interactive_output

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=df_pnl["dates"], y=df_pnl["cumpnl"], data=df_pnl, ax=ax)
    fig.show()
    '''
    import plotly.graph_objs as go
    
    # 绘制折线图
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_pnl['dates'], y=df_pnl['cumpnl'], mode='lines'))

    # 添加滚轮缩放
    fig.update_layout(title="%s PnL"%(alpha_name), xaxis_title='Date', yaxis_title='PnL')
    fig.update_xaxes(rangeslider_visible=True)
    
    # 设置画布大小
    fig.update_layout(
        title="%s PnL"%(alpha_name),
        xaxis_title='Date',
        yaxis_title='PnL',
        width=1024,
        height=680
    )

    # 显示图像
    fig.show()

def visual_summary(pnl_path):
    visualize_pnl(pnl_path)
    simsummary(pnl_path)