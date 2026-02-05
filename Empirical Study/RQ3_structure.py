import glob
from Attention import DeepAttentionAnalyzer
from util.remove_comments import remove_comments_and_docstrings
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
import os


def process_and_save_scores():
    # ================= 配置路径 =================
    data_dir = './data'
    input_csv_path = './RQ3/golden_cluster_3_with_attention.csv'
    output_dir = './RQ3'
    output_csv_path = os.path.join(output_dir, 'golden_cluster_3_with_attention.csv')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ================= 1. 加载 CSV 和 目标索引 =================
    print(f"Loading CSV from {input_csv_path}...")
    try:
        # 读取原始 CSV
        df = pd.read_csv(input_csv_path)
        target_indices_map = {idx: row_id for row_id, idx in enumerate(df.iloc[:, 0])}
        target_indices_set = set(target_indices_map.keys())

        print(f"Target sample count: {len(target_indices_set)}")

    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    code_summary_map = {}

    jsonl_files = glob.glob(os.path.join(data_dir, "filtered_python_train_*.jsonl"))
    # 按文件名数字排序
    try:
        jsonl_files.sort(key=lambda x: int(x.split('filtered_python_train_')[-1].split('.jsonl')[0]))
    except ValueError:
        jsonl_files.sort()

    global_idx = 0
    max_target_idx = max(target_indices_set) if target_indices_set else 0

    print("Scanning JSONL files to retrieve source code...")

    with tqdm(total=max_target_idx + 1, desc="Scanning Data", unit="line") as pbar:
        for file_path in jsonl_files:
            if global_idx > max_target_idx: break

            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if global_idx > max_target_idx: break
                    if global_idx in target_indices_set:
                        try:
                            js = json.loads(line)
                            code = remove_comments_and_docstrings(source=js['code'], lang='python')
                            summary = ' '.join(js['cleaned_docstring_tokens'])
                            code_summary_map[global_idx] = (code, summary)
                        except:
                            pass
                    global_idx += 1
                    pbar.update(1)

    print(f"Retrieved {len(code_summary_map)} code snippets.")

    analyzer = DeepAttentionAnalyzer()
    ast_sequence_column = ["[]"] * len(df)

    print("Calculating Attention Scores and Formatting...")

    for global_idx, (code, summary) in tqdm(code_summary_map.items(), desc="BERT Inference"):
        try:
            _, _, _, st_data = analyzer.analyze(code, summary)

            if not st_data:
                continue
            formatted_list = []
            for stmt in st_data:
                formatted_list.append({
                    "type": stmt['type'],
                    "score": round(stmt['score'], 4)  # 保留4位小数
                })
            json_str = json.dumps(formatted_list)
            row_id = target_indices_map[global_idx]
            ast_sequence_column[row_id] = json_str

        except Exception as e:
            print(f"Error analyzing index {global_idx}: {e}")
            pass

    print("Updating DataFrame...")
    df['AST_Sequence'] = ast_sequence_column

    print(f"Saving to {output_csv_path}...")
    df.to_csv(output_csv_path, index=False)
    print("Done!")

# ================= 配置 =================
INPUT_CSV = './RQ3/golden_cluster_3_with_attention.csv'
OUTPUT_CSV = './RQ3/golden_cluster_3_final_layer_stats_extended.csv'

# 映射字典
LAYER_MAPPING = {
    # ================= 1. 签名与接口层 (Interface Layer) =================
    # 定义“我是谁”以及“我产出什么”
    "FunctionDef": "Interface",
    "AsyncFunctionDef": "Interface",
    "ClassDef": "Interface",
    "Return": "Interface",  # 标志着接口调用的结束和产出
    "Yield": "Interface",  # 生成器接口产出
    "YieldFrom": "Interface",

    # ================= 2. 控制流层 (Control Flow Layer) =================
    # 决定代码的“走向”和“分支”
    "If": "Control",
    "Elif": "Control",  # 虚拟节点
    "Else": "Control",  # 虚拟节点
    "For": "Control",
    "AsyncFor": "Control",
    "While": "Control",
    "Try": "Control",
    "ExceptHandler": "Control",
    "With": "Control",  # 上下文管理通常涉及资源生命周期控制
    "AsyncWith": "Control",
    "Break": "Control",  # 改变循环流向
    "Continue": "Control",  # 改变循环流向
    "Raise": "Control",  # 显式中断流向（抛错）
    "Match": "Control",  # Python 3.10+ 模式匹配
    "match_case": "Control",  # 虚拟或实际节点（取决于实现）

    # ================= 3. API 序列层 (API Sequence Layer) =================
    # 核心业务逻辑：负责“调用谁”和“执行动作”
    # 这是我们刚刚通过 StatementSplitter 细分出来的类型
    "MethodCall": "API",  # 独立调用: obj.action()
    "FunctionCall": "API",  # 独立调用: print()
    "AwaitCall": "API",  # 异步调用: await func()

    # 带有赋值的调用，虽然改变了数据，但其核心目的是执行动作
    # 例如: response = client.get() -> 重点是 get 动作，而非 response 变量
    "AssignMethod": "API",  # 赋值调用: x = obj.action()
    "AssignFunc": "API",  # 赋值调用: x = calculate()

    # ================= 4. 数据流层 (Data Flow Layer) =================
    # 负责内部状态管理：涉及纯计算、变量引用、模块引入
    "Assign": "Data",  # 纯赋值: x = 1, x = a + b (剥离了函数调用后)
    "AnnAssign": "Data",  # 类型注释赋值: x: int = 1
    "AugAssign": "Data",  # 增量赋值: x += 1

    # 在剥离了 Call 之后，剩下的 Expr 通常是 Docstring 或无副作用的表达式
    "Expr": "Data",  # 主要是 Docstrings ("""...""") 或被忽略的计算

    "Import": "Data",  # 引入依赖（静态数据上下文）
    "ImportFrom": "Data",
    "Delete": "Data",  # 状态销毁: del x
    "Assert": "Data",  # 数据校验（虽然若失败会抛错，但语义上是验证数据）
    "Global": "Data",  # 作用域修改
    "Nonlocal": "Data",  # 作用域修改
    "Pass": "Data",  # 占位符

}


def get_row_metrics(row_json):
    try:
        stmt_list = json.loads(row_json)
    except:
        return None

    if not stmt_list:
        return None

    # 1. 统计
    layer_scores = {"Interface": 0.0, "Control": 0.0, "Data": 0.0, "API": 0.0}
    layer_counts = {"Interface": 0.0, "Control": 0.0, "Data": 0.0, "API": 0.0}
    total_score = 0.0
    total_count = 0

    for stmt in stmt_list:
        sType = stmt.get('type', 'Data')
        score = stmt.get('score', 0.0)
        layer = LAYER_MAPPING.get(sType, "Data")

        layer_scores[layer] += score
        layer_counts[layer] += 1
        total_score += score
        total_count += 1

    if total_score == 0 or total_count == 0:
        return None

    metrics = {}
    global_avg = total_score / total_count

    for layer in ["Interface", "Control", "Data", "API"]:
        l_score = layer_scores[layer]
        l_count = layer_counts[layer]

        # Metric A: Attention Share (注意力占比)
        metrics[f"{layer}_AttnShare"] = l_score / total_score

        # Metric B: Physical Share (物理数量占比) -> [新增]
        # 这一层的语句数量占总语句数量的百分比
        metrics[f"{layer}_CountShare"] = l_count / total_count

        # Metric C: Intensity (相对强度)
        if l_count > 0 and global_avg > 0:
            metrics[f"{layer}_Intensity"] = (l_score / l_count) / global_avg
        else:
            metrics[f"{layer}_Intensity"] = None

    return metrics


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    print("Reading data...")
    df = pd.read_csv(INPUT_CSV)

    # 容器
    agg_data = {
        "Interface": {"attn_shares": [], "count_shares": [], "intensities": []},
        "Control": {"attn_shares": [], "count_shares": [], "intensities": []},
        "Data": {"attn_shares": [], "count_shares": [], "intensities": []},
        "API": {"attn_shares": [], "count_shares": [], "intensities": []}
    }

    print("Aggregating statistics...")
    for json_str in tqdm(df['AST_Sequence']):
        res = get_row_metrics(json_str)
        if res:
            for layer in ["Interface", "Control", "Data", "API"]:
                agg_data[layer]["attn_shares"].append(res[f"{layer}_AttnShare"])
                agg_data[layer]["count_shares"].append(res[f"{layer}_CountShare"])  # 新增

                if res[f"{layer}_Intensity"] is not None:
                    agg_data[layer]["intensities"].append(res[f"{layer}_Intensity"])

    # === 计算最终平均值 ===
    final_rows = []

    print("\nCalculating final averages...")
    for layer in ["Interface", "Control", "Data", "API"]:
        # 1. Avg Attention Share
        avg_attn = np.mean(agg_data[layer]["attn_shares"])

        # 2. Avg Count Share (Physical Volume)
        avg_count = np.mean(agg_data[layer]["count_shares"])

        # 3. Avg Intensity
        avg_int = np.mean(agg_data[layer]["intensities"]) if agg_data[layer]["intensities"] else 0.0

        # 4. [高级指标] Attention Gain (注意力增益系数)
        # Gain > 1 表示模型放大了这一层的重要性
        # Gain < 1 表示模型压缩/忽略了这一层
        gain = avg_attn / avg_count if avg_count > 0 else 0.0

        final_rows.append({
            "Layer": layer,
            "Avg_Intensity": round(avg_int, 4),
            "Avg_Attn_Share": round(avg_attn, 4),  # 注意力占比
            "Avg_Count_Share": round(avg_count, 4),  # 物理数量占比
            "Gain_Factor": round(gain, 2)  # 增益系数
        })

    # === 保存 ===
    result_df = pd.DataFrame(final_rows)
    # 调整列顺序
    result_df = result_df[["Layer", "Avg_Intensity", "Avg_Attn_Share", "Avg_Count_Share", "Gain_Factor"]]

    print(f"\nFinal Result:\n{result_df.to_string(index=False)}")
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved extended metrics to {OUTPUT_CSV}")



if __name__ == "__main__":
    # process_and_save_scores()
    main()
