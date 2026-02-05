# def normalize_dicts(dict_list):
#     """
#     对多个字典中的元素进行归一化（min-max 到 0~1）
#     :param dict_list: 字典列表，例如 [{"a": 3, "b": 5}, {"a": 6, "c": 2}]
#     :return: {key: [归一化后的值数组]}
#     """
#     from collections import defaultdict
#
#     # 收集所有key的值
#     values_dict = defaultdict(list)
#     for d in dict_list:
#         for key, value in d.items():
#             values_dict[key].append(value)
#
#     # 对每个key归一化
#     normalized = {}
#     for key, values in values_dict.items():
#         min_v, max_v = min(values), max(values)
#         if min_v == max_v:  # 避免除零
#             norm_values = [1.0 for _ in values]
#         else:
#             norm_values = [(v - min_v) / (max_v - min_v) for v in values]
#         normalized[key] = norm_values
#
#     return normalized
#
#
# # 示例
# dicts = [
#     {'BLEU': 6.6999, 'BLEU-CN': 13.5162, 'METEOR': 34.9379, 'BERTSCORE': 62.272, 'SIDESCORE': 84.4935,
#      'ROUGE-L': 28.5286, 'Appropriateness': 3.8237},
#     {'BLEU': 9.2956, 'BLEU-CN': 20.4877, 'METEOR': 32.8622, 'BERTSCORE': 65.4402, 'SIDESCORE': 83.3812,
#      'ROUGE-L': 35.2746, 'Appropriateness': 4.6266},
#     {'BLEU': 8.4712, 'BLEU-CN': 18.8045, 'METEOR': 33.5524, 'BERTSCORE': 65.6532, 'SIDESCORE': 83.964,
#      'ROUGE-L': 33.4756, 'Appropriateness': 4.1891},
#     {'BLEU': 6.5324, 'BLEU-CN': 13.8271, 'METEOR': 33.9512, 'BERTSCORE': 62.5841, 'SIDESCORE': 43.1536,
#      'ROUGE-L': 28.5637, 'Appropriateness': 3.8839},
#     {'BLEU': 8.3602, 'BLEU-CN': 18.5164, 'METEOR': 31.743, 'BERTSCORE': 64.7559, 'SIDESCORE': 83.1397,
#      'ROUGE-L': 32.3523, 'Appropriateness': 4.1309},
#     {'BLEU': 8.2175, 'BLEU-CN': 18.3487, 'METEOR': 30.7184, 'BERTSCORE': 64.1188, 'SIDESCORE': 83.3977,
#      'ROUGE-L': 31.9162, 'Appropriateness': 3.9829}
# ]
#
# result = normalize_dicts(dicts)
# print(result)
# 输出示例:
# {
#   'a': [0.0, 1.0],
#   'b': [1.0],
#   'c': [0.0]
# }


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#
# # 输入数据
# x = [1, 2, 3, 4, 5, 6]
#
# # bleu = [0.000000, 0.548495, 0.836120, 0.989967, 1.000000, 0.946488]
# # bleu_cn = [0.000000, 0.570210, 0.857451, 1.000000, 0.998906, 0.973262]
# # meteor = [1.000000, 0.973859, 0.168998, 0.442278, 0.263247, 0.000000]
# # rouge = [0.000000, 0.642197, 0.893345, 0.981755, 1.000000, 0.983313]
# # bert = [0.000000, 0.680060, 0.916550, 0.994998, 1.000000, 0.990000]
# # g_eval = [0.0, 0.6523900948139619, 0.9287789418633368, 1.0, 0.9345998509782117, 0.9961463544036873]
#
#
#
# bleu = [0.06061812391430241, 1.0, 0.7016502605674578, 0.0, 0.6614794441227564, 0.6098364215402429]
# bleu_cn = [0.0, 1.0, 0.7585598508212007, 0.04459585455067061, 0.7172344545650148, 0.6931793731621604]
# meteor = [1.0, 0.5080696765019558, 0.6716435596634671, 0.7661571276217564, 0.24282497926294572, 0.0]
# rouge = [0.0, 1.0, 0.7333234509338867, 0.005203083308627322, 0.566809961458642, 0.5021642454788022]
# bert = [0.0, 0.9370046137466006, 1.0, 0.09230450727552376, 0.7346208446705308, 0.546196616585826]
# g_eval = [0.0, 1.0, 0.4551002615518743, 0.07497820401046214, 0.38261302777431755, 0.19828123053929486]
#
# # 设置颜色代码
# color1 = "#038355"  # 孔雀绿
# color2 = "#ffc34e"  # 向日黄
#
# # 设置字体
# font = {'family': 'Times New Roman',
#         'size': 12}
# plt.rc('font', **font)
#
# plt.figure(figsize=(6.4, 4.8))
#
# plt.plot(x, bleu, color='red', label='BLEU', linewidth=1.5)
# plt.plot(x, bleu_cn, color='orange', label='BLEU-CN', linewidth=1.5)
# plt.plot(x, meteor, color='gold', label='METEOR', linewidth=1.5)
# plt.plot(x, rouge, color='green', label='ROUGE-L', linewidth=1.5)
# plt.plot(x, bert, color='blue', label='BERTScore', linewidth=1.5)
# plt.plot(x, g_eval, color='black', label='G-eval', linewidth=1.5)
#
# plt.xlim(1, 6)
# plt.ylim(0, 1)
#
# plt.xticks([1, 2, 3, 4, 5, 6], labels=[5, 10, 25, 50, 75, 100])  # x轴标签换成 0-5
# # plt.xticks([1, 2, 3, 4], labels=[5,10,50,100])  # x轴标签换成 0-5
# plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
#
# # plt.xlabel("Maximum Score in Validator " + r"$s_{max}$")
# plt.xlabel(r"$T_{max}$")
# plt.ylabel("Normalized Value")
#
# plt.grid(True, linestyle='--', alpha=0.7)
#
# plt.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1, 0.98))
#
# # plt.savefig("t_max_effectiveness.pdf", dpi=300, bbox_inches='tight')
# plt.savefig("s_max_effectiveness.pdf", dpi=300, bbox_inches='tight')
#
# plt.show()

##########################################################################################

import matplotlib.pyplot as plt
import numpy as np

# 示例数据

# x = np.arange(0, 6)  # x 轴：迭代次数 (0~5)
# bar_data = np.array([
#     [30, 30, 30, 30, 30, 30],  # 迭代0通过
#     [0, 72, 72, 72, 72, 72],  # 迭代1通过
#     [0, 0, 43, 43, 43, 43],  # 迭代2通过
#     [0, 0, 0, 25, 25, 25],  # 迭代3通过
#     [0, 0, 0, 0, 10, 10],  # 迭代4通过
#     [0, 0, 0, 0, 0, 20],  # 迭代5通过
# ])
#
# # 设置字体
# font = {'family' : 'Times New Roman',
#         'size'   : 12}
# plt.rc('font', **font)
#
# # 堆叠柱状图
#
# labels = ["0", "1", "2", "3", "4", "5"]
# colors = ["#5DC1B9", "#F5866A", "#648FFF", "#DC71C0", "#82C341", "#FFD700"]
#
# bottom = np.zeros(len(x))
# for i, (label, color) in enumerate(zip(labels, colors)):
#     plt.bar(x, bar_data[i], bottom=bottom, label=f"iter {label}", color=color, width=0.5)
#     bottom += bar_data[i]
#
# # # 在 x=5 的位置额外加一个颜色块 (高度=18)
# # plt.bar(5, 18, bottom=bottom[5], color="#bcc4bf", label="extra block", width=0.5)
#
# # 折线数据（累计成功样本数）
#
# line_y = np.array([30, 102, 145, 170, 180, 182])
# plt.plot(x, line_y, color="blue", marker="o")
#
# # 在折线上标注数值
#
# for i, val in enumerate(line_y):
#     plt.text(x[i], val + 0.5, str(val), ha='center', va='bottom', fontsize=9)
#
# # 坐标轴和标签
#
# plt.xlabel("Number of Iterations "+"$T_{max}$")
# plt.ylabel("Number of Comments")
# plt.xticks(x)
# plt.ylim(0, 200)
#
# # 图例
#
# plt.legend(loc="upper right", bbox_to_anchor=(1.25, 0.5), title="Pass at iter")
#
# plt.tight_layout()
#
# plt.savefig("iters_pass.pdf", dpi=300, bbox_inches='tight')
#
# plt.show()


###########################################################################################
import matplotlib.pyplot as plt
import numpy as np
#
# # 示例数据
#
# x = np.arange(0, 4)  # x 轴：迭代次数 (0~5)
# bar_data = np.array([
#     [110, 21, 91, 0],
#     [68, 93, 87, 2],
#     [14, 33, 16, 6],
#     [8, 53, 6, 192]
# ])
from adjustText import adjust_text
def autopct_format(pct):
    """自定义百分比显示函数：占比为0时不显示"""
    return ('%1.1f%%' % pct) if pct > 0 else ''

data_list = [
    [111, 68, 14, 7],
    [20, 93, 33, 54],
    [17, 61, 46, 76],
    [91, 87, 16, 6],
    [4, 38, 24, 134],
    [0, 2, 6, 192]
]
labels = ['iter 0', 'iter 1', 'iter 2', 'iter 3']
s = ['5', '10', '25', '50', '75', '100']
colors = ["#5DC1B9", "#F5866A", "#648FFF", "#DC71C0", "#82C341", "#FFD700"]

font = {'family' : 'Times New Roman',
        'size'   : 15}
plt.rc('font', **font)

# 你之前定义的百分比格式化函数，假设是这样
def autopct_format(pct):
    return f"{pct:.1f}%" if pct > 0 else ""

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for i, data in enumerate(data_list):
    row, col = divmod(i, 3)
    ax = axes[row][col]

    wedges, texts, autotexts = ax.pie(
        data,
        labels=[None]*len(labels),  # labels 不显示在饼图上
        colors=colors,
        autopct=autopct_format,
        startangle=90
    )

    # 收集文本（标签 + 百分比）
    all_texts = autotexts

    # 自动调整，避免重叠
    adjust_text(all_texts, ax=ax,)

    ax.set_title("Smax=" + s[i])

plt.tight_layout()
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 0.5), title="Pass at iter", labels=labels)


plt.savefig("smax_pass.pdf", dpi=300, bbox_inches='tight')
plt.show()


##############################训练集G-eval#########################################
# import matplotlib.pyplot as plt
#
# x = [1, 2, 3, 4, 5]  # x轴
# # values = [818, 3262, 2121, 3564, 235]
# values = [1280, 2443, 1946, 3163, 1168]
#
# font = {'family' : 'Times New Roman',
#         'size'   : 12}
# plt.rc('font', **font)
#
# plt.bar(x, values, color='#5DC1B9')
#
# # plt.xlabel("G-eval score for python train set")
# plt.xlabel("G-eval score for java train set")
# plt.ylabel("Number of comments")
#
# # 固定 y 轴范围
# plt.ylim(0, 5000)
#
# # 显示每个柱子的数值
# for i, v in enumerate(values):
#     plt.text(x[i], v + 0.5, str(v), ha='center', va='bottom', fontsize=9)
#
# # plt.savefig("python_valid.pdf", dpi=300, bbox_inches='tight')
# plt.savefig("java_valid.pdf", dpi=300, bbox_inches='tight')
#
# plt.show()

#####################################################################
#
# import numpy as np
# from scipy.stats import normaltest
# from scipy.stats import shapiro
#
# data = [0,16,1178,767,1448,1139,1383,1844,1507,718]
# values1 = [818, 3262, 2121, 3564, 235]
# values2 = [1280, 2443, 1946, 3163, 1168]
#
# stat, p = normaltest(data)
# print("statistic:", stat)
# print("p-value:", p)
#
# if p < 0.05:
#     print("拒绝原假设：数据不是正态分布")
# else:
#     print("不能拒绝原假设：数据可能是正态分布")
#
#
# print(shapiro(values1))
# print(shapiro(values2))