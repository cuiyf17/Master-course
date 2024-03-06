import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import cv2

RANDOM_TEXT = ""

import numpy as np
import cuiyf_op.cuiyfOp as OP


import matplotlib
def plot_group_ret(group_pnl_path):
    """
    Plot pnl of a group of alphas
    """
    df = pd.read_csv(group_pnl_path, sep=" ", header = None)
    dates = pd.to_datetime(df.iloc[:,0], format="%Y%m%d")
    cmap = matplotlib.colormaps["Spectral"]
    colors = [cmap(i/(df.shape[1]-2)) for i in range(df.shape[1]-1)]
    plt.figure(figsize=(10,5))
    sns.set(style="whitegrid")  # 设置带有网格的背景
    zz500 = df.iloc[:,df.shape[1]-1].cumsum()
    for i in range(1, df.shape[1] - 1):
        sns.lineplot(x=dates, y=df.iloc[:,i].cumsum()-zz500, color = colors[i], label=str(i))  # 使用Seaborn的lineplot函数
    plt.legend()
    plt.title("Group Excess Cumulative Returns (against zz500)")
    plt.show()

def plot_group_pnl(group_pnl_path):
    """
    Plot pnl of a group of alphas
    """
    df = pd.read_csv(group_pnl_path, sep=" ", header = None)
    dates = pd.to_datetime(df.iloc[:,0], format="%Y%m%d")
    cmap = matplotlib.colormaps["Spectral"]
    colors = [cmap(i/(df.shape[1]-1)) for i in range(df.shape[1])]
    plt.figure(figsize=(10,5))
    sns.set(style="whitegrid")  # 设置带有网格的背景
    for i in range(1, df.shape[1]):
        sns.lineplot(x=dates, y=df.iloc[:,i].cumsum(), color = colors[i], label=str(i))  # 使用Seaborn的lineplot函数
    plt.legend()
    plt.title("Group Cumulative PnL")
    plt.show()

def plot_longshort_pnl(pnl_path):
    df = pd.read_csv(pnl_path, sep=" ", header = None)
    dates = pd.to_datetime(df.iloc[:,0], format="%Y%m%d")
    plt.figure(figsize=(10,5))
    sns.set(style="whitegrid")  # 设置带有网格的背景
    sns.lineplot(x=dates, y=df.iloc[:,1].cumsum(), color = "red", label="long")  # 使用Seaborn的lineplot函数
    sns.lineplot(x=dates, y=df.iloc[:,2].cumsum(), color = "blue", label="short")  # 使用Seaborn的lineplot函数
    plt.legend()
    plt.title("Long Short PnL")
    plt.show()

def checkcorr(pnl_path):
    columns = ["dates", "pnl", "long", "short", "sh_hold", "sh_trd", "avg_ret", "b_sh", "t_sh", "xsy", "xsyy"]
    df_pnl = pd.read_csv(pnl_path, sep=" ", header=None)
    df_pnl.columns = columns
    df_pnl["dates"] = pd.to_datetime(df_pnl["dates"], format="%Y%m%d")
    
    cmp_folder = "/data/share/poolpnl/"
    files = [os.path.join(cmp_folder, x) for x in os.listdir(cmp_folder)]
    
    bcorrs = dict()
    for i in range(19, -1, -1):
        start = (i - 10)/10
        bcorrs[start] = [0,0,0]

    for i, file in enumerate(files):
        cmp_pnl = pd.read_csv(file, sep=" ", header=None)
        cmp_pnl.columns = ["dates", "pnl"]
        isintraday = False
        if(":" in str(cmp_pnl["dates"][0])):
            continue
            cmp_pnl["dates"] = (pd.to_datetime(cmp_pnl["dates"], format="%Y%m%d-%H:%M:%S").dt.date).astype("datetime64[ns]")
            isintraday = True
        else:
            try:
                cmp_pnl["dates"] = pd.to_datetime(cmp_pnl["dates"], format="%Y%m%d")
            except:
                print(cmp_pnl["dates"])
                continue

        start_time = max(cmp_pnl["dates"][0], df_pnl["dates"][0])
        end_time = min(cmp_pnl["dates"][len(cmp_pnl) - 1], df_pnl["dates"][len(df_pnl) - 1])

        cmp_pnl = cmp_pnl[(cmp_pnl["dates"] >= start_time) & (cmp_pnl["dates"] <= end_time)]
        if(isintraday):  
            cmp_pnl_sum = cmp_pnl.groupby(cmp_pnl["dates"])["pnl"].transform("sum")
        else:
            cmp_pnl_sum = cmp_pnl["pnl"]
        tmp_pnl = df_pnl[(df_pnl["dates"] >= start_time) & (df_pnl["dates"] <= end_time)]
        bcorr = np.corrcoef(tmp_pnl["pnl"], cmp_pnl_sum)[0][1]
        df_sharpe = tmp_pnl["pnl"].mean()/tmp_pnl["pnl"].std()
        cmp_sharpe = cmp_pnl_sum.mean()/cmp_pnl_sum.std()

        idx = int(bcorr*10)/10
        bcorrs[idx][0] += 1
        if(df_sharpe >= cmp_sharpe):
            bcorrs[idx][1] += 1
        else:
            bcorrs[idx][2] += 1

    print("%-7s, %-5s, %-5s, %-5s"%("corr", "count", "better", "worse"))
    for item in bcorrs:
        start = item
        end = item + 0.1
        print("%.1f-%.1f: %-5d, %-5d, %-5d"%(start, end, bcorrs[item][0], bcorrs[item][1], bcorrs[item][2]))

def simsummary(pnl_path):
    os.system('simsummary ' + pnl_path)

def bcorr(pnl_path, target_folder, isprint = True, show_num = 0, basic_barra = False, TopN = 0):
    if(type(target_folder) == str):
        target_folder = [target_folder]
    columns = ["pnl", "long", "short", "sh_hold", "sh_trd", "avg_ret", "b_sh", "t_sh", "xsy", "xsyy"]
    df_pnl = pd.read_csv(pnl_path, sep=" ", header=None, index_col = 0)
    df_pnl.columns = columns
    roots = [os.path.abspath(x) for x in target_folder]
    if(basic_barra):
        roots = [os.path.abspath(x) for x in roots if("Basic" in x or "Barra" in x)]
    ret_str = ""
    bcorr_lists = []
    imax_len = 0
    for root in roots:
        #if(isprint):
        #    print(root + ":")
        #else:
        target_pnls = [x for x in os.listdir(root) if not (x.endswith('.xml') or x.endswith('.csv') or x.endswith('.txt') or x.endswith('.pkl') or x.endswith('.ipynb') or x.endswith('.py') or x.endswith('.checkpoint'))]
        max_len = [len(x) for x in target_pnls]
        imax_len = max(max(max_len), imax_len)
        bcorr_list = []
        for name in target_pnls:
            target_pnl = os.path.join(root, name)
            df_terget_pnl = pd.read_csv(target_pnl, sep=" ", header=None, index_col = 0)
            df_terget_pnl.columns = columns
            corr = df_pnl["pnl"].corr(df_terget_pnl["pnl"])
            bcorr_list.append((name, corr))
            #os.system("python /home/cuiyf/myalphasim/cuiyf_op/bcorr.py " + pnl_path + " " + target_pnl + " 2 2")
        bcorr_list.sort(key = lambda x: x[1], reverse = True)
        bcorr_lists.append(bcorr_list)
    if(TopN > 0):
        bcorr_all = []
        for bcorr_list in bcorr_lists:
            bcorr_all.extend(bcorr_list)
        bcorr_all.sort(key = lambda x: x[1])
        bcorr_all = bcorr_all[-TopN:]
        ret_str += "Top %d corr of my alphas:\n"%(TopN)
        for name, corr in bcorr_all:
            ret_str += ": %.6f  "%(corr) + "%s"%(name) + "\n"
    else:
        for i, root in enumerate(roots):
            lit = bcorr_lists[i]
            ret_str += root + ":\n"
            if(show_num == 0):
                for name, corr in lit:
                    tmp_str = "%-" + str(imax_len) + "s"
                    ret_str += "    " + tmp_str%(name) + ": %.6f\n"%(corr)
            else:
                for i, (name, corr) in enumerate(lit):
                    if(i >= show_num):
                        break
                    tmp_str = "%-" + str(imax_len) + "s"
                    ret_str += "    " + tmp_str%(name) + ": %.6f\n"%(corr)
    if(not isprint):
        return ret_str
    else:
        print(ret_str)

def corr_heatmap(pnl_folders):
    pnl_list = []
    for pnl_folder in pnl_folders:
        tmplist = [os.path.join(pnl_folder, x) for x in os.listdir(pnl_folder)]
        pnl_list.extend(tmplist)
    pnl_list.sort()
    corr_list = []
    for pnl_path in pnl_list:
        pnl = pd.read_csv(pnl_path, header = None, sep = " ").iloc[:, 1]
        corr_list.append(pnl)
    corr_list = np.array(corr_list)
    corr_matrix = np.corrcoef(corr_list).round(2)
    corr_matrix = pd.DataFrame(corr_matrix, index = [os.path.basename(x) for x in pnl_list], columns = [os.path.basename(x) for x in pnl_list])
    plt.figure(figsize = (15,15))
    sns.heatmap(corr_matrix, annot = True, vmax = 1, vmin = -1, square = True, cmap = "vlag", xticklabels=True, yticklabels=True)
    plt.title("Correlation Heatmap (%d alphas)"%(len(pnl_list)))
    plt.show()

def text_to_image(text, font_size=1):
    global RANDOM_TEXT
    # 加载字体
    font = cv2.FONT_HERSHEY_COMPLEX
    # 创建图像
    image = np.zeros((1024, 1024, 3), dtype=np.uint8) + 255
    text = text.split("\n")
    for i , txt in enumerate(text):
        # 计算文本大小
        text_width, text_height = cv2.getTextSize(txt, font, font_size, thickness=1)[0]
        # 计算文本位置
        x = 0 #(image.shape[1] - text_width)
        y = (text_height) + int(i * text_height * 1.7)
        # 绘制文本
        cv2.putText(image, txt, (x, y), font, font_size, (0, 0, 0), thickness=2)
    for filename in os.listdir("/home/cuiyf/myalphasim/cuiyf_op/"):
        if filename.endswith(".png"):
            os.remove(os.path.join("/home/cuiyf/myalphasim/cuiyf_op/", filename))
    RANDOM_TEXT = str(np.random.randint(100000000000))
    path = "/home/cuiyf/myalphasim/cuiyf_op/tmp" + RANDOM_TEXT + ".png"
    cv2.imwrite(path, image)

    return RANDOM_TEXT
  
def visual_report(pnl_path, cmp_paths = None):
    alpha_name = os.path.basename(pnl_path)
    columns = ["dates", "pnl", "long", "short", "sh_hold", "sh_trd", "avg_ret", "b_sh", "t_sh", "xsy", "xsyy"]
    df_pnl = pd.read_csv(pnl_path, sep=" ", header=None)
    df_pnl.columns = columns
    df_pnl["dates"] = pd.to_datetime(df_pnl["dates"], format="%Y%m%d")
    df_pnl["cumpnl"] = df_pnl["pnl"].cumsum()

    df_longshort_pnl = pd.read_csv(pnl_path.replace("pnl", "longshort_pnl"), sep=" ", header=None)
    df_longshort_pnl.columns = ["dates", "long", "short"]
    df_longshort_pnl["dates"] = pd.to_datetime(df_longshort_pnl["dates"], format="%Y%m%d")
    df_longshort_pnl["long"] = df_longshort_pnl["long"].cumsum()
    df_longshort_pnl["short"] = df_longshort_pnl["short"].cumsum()

    df_group_pnl = pd.read_csv(pnl_path.replace("pnl", "group_pnl"), sep=" ", header=None)

    df_group_ret = pd.read_csv(pnl_path.replace("pnl", "group_ret"), sep=" ", header=None)


    # text = ""
    # if(cmp_paths is not None):
    #     text = bcorr(pnl_path, cmp_paths, isprint=False, show_num=0, basic_barra=False, TopN=10)
    #     text = text_to_image(text)
    #     text = "/home/cuiyf/myalphasim/cuiyf_op/tmp" + text + ".png"
    
    fig= plt.figure(figsize=(15, 10))
    # 添加左上角的子图
    ax1 = fig.add_axes([0, 0.55, 0.35, 0.35])  # left, bottom, width, height
    # 添加右上角的子图
    ax2 = fig.add_axes([0.36, 0.55, 0.64, 0.35])
    # 添加左下方的子图
    ax3 = fig.add_axes([0, 0.1, 0.475, 0.35])
    # 添加右下方的子图
    ax4 = fig.add_axes([0.525, 0.1, 0.475, 0.35])
    ax1.plot(df_pnl['dates'], df_pnl['cumpnl'])
    ax1.grid(True)
    ax1.set_title('Cumulative PnL')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('PnL')
    ax1.legend(loc='upper left')

    os.system("python /home/cuiyf/myalphasim/cuiyf_op/cyfsimsummary.py %s > /home/cuiyf/myalphasim/cuiyf_op/simsummary.txt"%(pnl_path))
    simsummary_df = pd.read_csv("/home/cuiyf/myalphasim/cuiyf_op/simsummary.txt", sep="\s+", index_col=False)
    ax2.axis('off')
    cell_text = simsummary_df.values
    max_lengths = []
    for i in range(cell_text.shape[1]):
        max_lengths.append(max([len(str(x)) for x in cell_text[:, i]]))
    col_widths = [min(0.1 + max_length*0.05, 1) for max_length in max_lengths]
    table = ax2.table(cellText=simsummary_df.values, 
                      colLabels=simsummary_df.columns, 
                      colWidths=col_widths, 
                      cellLoc = 'center', 
                      loc='center', 
                      colLoc='center', 
                      bbox=[0, 0, 1, 1],
                      edges =  'open'# or {'open', 'closed', 'horizontal', 'vertical'}
                      )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    ax2.grid(True)
    ax2.set_title('SimSummary Result')

    ax3.plot(df_longshort_pnl['dates'], df_longshort_pnl['long'], color = "red", label="long")
    ax3.plot(df_longshort_pnl['dates'], df_longshort_pnl['short'], color = "blue", label="short")
    ax3.grid(True)
    ax3.set_title('Long&Short Cumulative Excess PnL (against zz500)')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('PnL')
    ax3.legend(loc='upper left')
    
    dates = pd.to_datetime(df_group_ret.iloc[:,0], format="%Y%m%d")
    cmap = matplotlib.colormaps["Spectral"]
    colors = [cmap(i/(df_group_ret.shape[1]-2)) for i in range(df_group_ret.shape[1]-1)]
    zz500 = df_group_ret.iloc[:,df_group_ret.shape[1]-1].cumsum()
    for i in range(1, df_group_ret.shape[1] - 1):
        ax4.plot(dates, df_group_ret.iloc[:,i].cumsum()-zz500, color = colors[i], label=str(i))
    ax4.grid(True)
    ax4.set_title('Group Cumulative Excess Returns (against zz500)')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Returns')
    ax4.legend(loc='upper left')

    fig.show()
    return

def ts_visual_report(pnl_path, cmp_paths = None):
    alpha_name = os.path.basename(pnl_path)
    columns = ["dates", "pnl", "long", "short", "sh_hold", "sh_trd", "avg_ret", "b_sh", "t_sh", "xsy", "xsyy"]
    df_pnl = pd.read_csv(pnl_path, sep=" ", header=None)
    df_pnl.columns = columns
    df_pnl["dates"] = pd.to_datetime(df_pnl["dates"], format="%Y%m%d")
    df_pnl["cumpnl"] = df_pnl["pnl"].cumsum()

    df_longshort_pnl = pd.read_csv(pnl_path.replace("pnl", "longshort_pnl"), sep=" ", header=None)
    df_longshort_pnl.columns = ["dates", "long", "short", "zz500 long", "zz500 short"]
    df_longshort_pnl["dates"] = pd.to_datetime(df_longshort_pnl["dates"], format="%Y%m%d")
    df_longshort_pnl["excess long"] = df_longshort_pnl["long"] - df_longshort_pnl["zz500 long"]
    df_longshort_pnl["excess short"] = df_longshort_pnl["short"] - df_longshort_pnl["zz500 short"]
    df_longshort_pnl["zz500"] = df_longshort_pnl["zz500 long"] + df_longshort_pnl["zz500 short"]
    

    # text = ""
    # if(cmp_paths is not None):
    #     text = bcorr(pnl_path, cmp_paths, isprint=False, show_num=0, basic_barra=False, TopN=10)
    #     text = text_to_image(text)
    #     text = "/home/cuiyf/myalphasim/cuiyf_op/tmp" + text + ".png"
    
    fig= plt.figure(figsize=(15, 10))
    # 添加左上角的子图
    ax1 = fig.add_axes([0, 0.55, 0.35, 0.35])  # left, bottom, width, height
    # 添加右上角的子图
    ax2 = fig.add_axes([0.36, 0.55, 0.64, 0.35])
    # 添加左下方的子图
    ax3 = fig.add_axes([0, 0.1, 0.475, 0.35])
    # 添加右下方的子图
    ax4 = fig.add_axes([0.525, 0.1, 0.475, 0.35])

    avg_long = df_pnl["long"].mean()
    and_short = df_pnl["short"].mean()
    sharpe = df_pnl["pnl"].mean()/df_pnl["pnl"].std() * np.sqrt(242)
    ax1.plot(df_pnl['dates'], df_pnl['cumpnl'], label="PnL = %.2f, Sharpe = %.2f\navg_long = %.2f\navg_short = %.2f"%(df_pnl['cumpnl'].iloc[-1], sharpe, avg_long, and_short))
    pnl = df_longshort_pnl["excess long"] - df_longshort_pnl["excess short"]
    sharpe = pnl.mean()/pnl.std() * np.sqrt(242)
    ax1.plot(df_pnl['dates'], pnl.cumsum(), label="Excess PnL, Sharpe = %.2f"%(sharpe))
    ax1.grid(True)
    ax1.legend()
    ax1.set_title('Cumulative PnL')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('PnL')
    ax1.legend(loc='upper left')

    os.system("python /home/cuiyf/myalphasim/cuiyf_op/cyfsimsummary.py %s > /home/cuiyf/myalphasim/cuiyf_op/simsummary.txt"%(pnl_path))
    simsummary_df = pd.read_csv("/home/cuiyf/myalphasim/cuiyf_op/simsummary.txt", sep="\s+", index_col=False)
    ax2.axis('off')
    cell_text = simsummary_df.values
    max_lengths = []
    for i in range(cell_text.shape[1]):
        max_lengths.append(max([len(str(x)) for x in cell_text[:, i]]))
    col_widths = [min(0.1 + max_length*0.05, 1) for max_length in max_lengths]
    table = ax2.table(cellText=simsummary_df.values, 
                      colLabels=simsummary_df.columns, 
                      colWidths=col_widths, 
                      cellLoc = 'center', 
                      loc='center', 
                      colLoc='center', 
                      bbox=[0, 0, 1, 1],
                      edges =  'open'# or {'open', 'closed', 'horizontal', 'vertical'}
                      )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    ax2.grid(True)
    ax2.set_title('SimSummary Result')

    sharpe = df_longshort_pnl["long"].mean()/df_longshort_pnl["long"].std() * np.sqrt(242)
    ax3.plot(df_longshort_pnl['dates'], df_longshort_pnl['long'].cumsum(), color = "red", label="Long, Sharpe = %.2f"%(sharpe))
    sharpe = df_longshort_pnl["short"].mean()/df_longshort_pnl["short"].std() * np.sqrt(242)
    ax3.plot(df_longshort_pnl['dates'], df_longshort_pnl['short'].cumsum(), color = "blue", label="Short, Sharpe = %.2f"%(sharpe))
    # ax3.plot(df_longshort_pnl['dates'], df_longshort_pnl['long'] - df_longshort_pnl['short'], color = "green", label="pnl")
    ax3.grid(True)
    ax3.set_title('Long&Short Cumulative PnL')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('PnL')
    ax3.legend(loc='upper left')
    
    sharpe = df_longshort_pnl["excess long"].mean()/df_longshort_pnl["excess long"].std() * np.sqrt(242)
    ax4.plot(df_longshort_pnl['dates'], df_longshort_pnl['excess long'].cumsum(), color = "red", label="Long, Sharpe = %.2f"%(sharpe))
    sharpe = df_longshort_pnl["excess short"].mean()/df_longshort_pnl["excess short"].std() * np.sqrt(242)
    ax4.plot(df_longshort_pnl['dates'], df_longshort_pnl['excess short'].cumsum(), color = "blue", label="Short, Sharpe = %.2f"%(sharpe))
    ax4.grid(True)
    ax4.set_title('Long&Short Cumulative Excess PnL (against zz500)')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('PnL')
    ax4.legend(loc='upper left')

    fig.show()
    return

def visualize_pnl(pnl_path, cmp_paths = None):
    alpha_name = os.path.basename(pnl_path)
    columns = ["dates", "pnl", "long", "short", "sh_hold", "sh_trd", "avg_ret", "b_sh", "t_sh", "xsy", "xsyy"]
    df_pnl = pd.read_csv(pnl_path, sep=" ", header=None)
    df_pnl.columns = columns
    df_pnl["dates"] = pd.to_datetime(df_pnl["dates"], format="%Y%m%d")
    df_pnl["cumpnl"] = df_pnl["pnl"].cumsum()

    text = ""
    if(cmp_paths is not None):
        text = bcorr(pnl_path, cmp_paths, isprint=False, show_num=1)
        text = text_to_image(text)
        text = "/home/cuiyf/myalphasim/cuiyf_op/tmp" + text + ".png"
    
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    
    # 绘制折线图
    # 创建两个子图
    # 设置画布大小
    fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3], horizontal_spacing=0.02)
    fig.update_layout(
        title="%s PnL"%(alpha_name),
        xaxis_title='Date',
        yaxis_title='PnL',
        width=1500,
        height=680,
    )
    fig.add_trace(go.Scatter(x=df_pnl['dates'], y=df_pnl['cumpnl'], mode='lines'), row=1, col=1)
    # 添加滚轮缩放
    fig.update_layout(title="%s PnL"%(alpha_name), xaxis_title='Date', yaxis_title='PnL')
    fig.update_xaxes(rangeslider_visible=True, row=1, col=1)
    fig.add_image( row=1, col=2)

    if(text != ""):
        # 显示图像
        fig.add_layout_image(
            dict(
                source=text,
                xref="x",
                yref="y",
                x=1.5,
                y=2.5,
                sizex=2,
                sizey=2,
                sizing="stretch",
                opacity=1.0,
                layer="below",
            ),
            row=1,
            col=2
        )
        fig.update_xaxes(showgrid=False, row=1, col=2)
        fig.update_yaxes(showgrid=False, row=1, col=2)

    # 显示图像
    fig.show()

    group_ret_path = pnl_path.replace("pnl", "group_ret")
    plot_group_ret(group_ret_path)

    group_pnl_path = pnl_path.replace("pnl", "group_pnl")
    plot_group_pnl(group_pnl_path)

    longshort_pnl_path = pnl_path.replace("pnl", "longshort_pnl")
    plot_longshort_pnl(longshort_pnl_path)



def visual_summary(pnl_path, cmp_paths = None):
    visualize_pnl(pnl_path, cmp_paths)
    simsummary(pnl_path)
