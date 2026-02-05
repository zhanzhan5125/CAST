import pandas as pd
import glob
import os

# ================= 配置区域 =================
# 这里请修改为你实际的CSV文件路径，例如 "./features/*.csv"
CSV_PATTERN = "../RQ2/golden_cluster_3_data.csv"
TARGET_COL = "AST_Sequence"  # 截断点列名


def truncate_csv_columns():
    files = glob.glob(CSV_PATTERN)

    if not files:
        print(f"未找到匹配的文件: {CSV_PATTERN}")
        return

    print(f"找到 {len(files)} 个文件，准备处理...")

    for file_path in files:
        try:
            # 读取CSV
            df = pd.read_csv(file_path)

            # 检查目标列是否存在
            if TARGET_COL not in df.columns:
                print(f"[跳过] {os.path.basename(file_path)} - 未找到列 '{TARGET_COL}'")
                continue

            # 获取目标列的索引位置
            # get_loc 返回的是整数索引
            col_index = df.columns.get_loc(TARGET_COL)

            # 截取：保留从第0列到目标列（包含目标列）
            # iloc[:, :col_index + 1] 表示：所有行，列从开头切片到 col_index+1 (不包含end，所以+1)
            df_new = df.iloc[:, :col_index + 1]

            # 覆盖保存原文件
            df_new.to_csv(file_path, index=False)

            # 计算删除了多少列
            removed_count = len(df.columns) - len(df_new.columns)
            print(f"[完成] {os.path.basename(file_path)} - 删除了后方 {removed_count} 列")

        except Exception as e:
            print(f"[错误] 处理 {file_path} 时出错: {e}")


def calculate_num():
    csv_files = glob.glob("../RQ2/golden_cluster_3_data.csv")  # 修改为你的路径
    total_rows = 0

    for file in csv_files:
        df = pd.read_csv(file)
        total_rows += len(df)

    print(f"CSV 文件数量: {len(csv_files)}")
    print(f"数据总行数: {total_rows}")


import pandas as pd
import glob
import os


def remove_column_batch(target_col='AST_Sequence', file_pattern='*.csv'):
    """
    批量删除当前目录下所有 CSV 文件中的指定列。
    策略：在读取时直接跳过该列，避免将巨大的 AST 数据加载到内存中。
    """
    # 获取当前目录下所有的 csv 文件
    files = glob.glob(file_pattern)

    if not files:
        print("当前目录下没有找到 CSV 文件。")
        return

    print(f"找到 {len(files)} 个文件，准备处理...")

    for file_path in files:
        try:
            # 1. 先只读一行，检查该列是否存在 (避免报错)
            peek_df = pd.read_csv(file_path, nrows=1)
            if target_col not in peek_df.columns:
                print(f"[跳过] {file_path} (不存在列 {target_col})")
                continue

            # 2. 核心技巧：使用 usecols 排除掉目标列
            # lambda c: c != target_col  的意思是：只要列名不是目标列，就读取
            print(f"[处理中] {file_path} ...", end="")

            # 使用 utf-8-sig 防止中文乱码，engine='c' 速度更快
            df = pd.read_csv(file_path, usecols=lambda c: c != target_col, encoding='utf-8')

            # 3. 覆盖写入原文件 (使用 quote_all 增加安全性)
            # index=False 假设原始文件已经包含了索引列，或者你不想要 Pandas 自动生成的索引
            df.to_csv(file_path, index=False, encoding='utf-8-sig')

            print(f" -> 成功 (剩余列数: {df.shape[1]})")

        except Exception as e:
            print(f"\n[错误] 处理 {file_path} 时失败: {e}")



if __name__ == "__main__":
    # truncate_csv_columns()
    # calculate_num()
    remove_column_batch()