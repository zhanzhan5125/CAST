import ast
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from AST import StatementSplitter


class DeepAttentionAnalyzer:
    def __init__(self, model_name="microsoft/codebert-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _map_code_subwords_to_ast(self, bert_offsets, ast_tokens):
        """
        Code 侧映射：将 Subwords 归并到 AST Tokens
        返回: List of [AST_Token_Obj, [subword_indices]]
        """
        mapped_data = []
        i = 0
        while i < len(bert_offsets):
            s_start, s_end = bert_offsets[i]
            s_center = (s_start + s_end) / 2

            matched_token = None
            for token in ast_tokens:
                if token.start_idx <= s_center < token.end_idx:
                    matched_token = token
                    break

            if matched_token:
                if not mapped_data or mapped_data[-1][0] != matched_token:
                    mapped_data.append([matched_token, []])
                mapped_data[-1][1].append(i)
            i += 1
        return mapped_data

    def _map_summary_subwords_to_words(self, bert_offsets, summary_text):
        """
        Summary 侧映射：将 Subwords 归并到自然单词 (split by space)
        返回: List of [Word_String, [subword_indices]]
        """
        # 1. 按照人类习惯切分单词
        words = summary_text.split()

        # 2. 计算每个单词在 Summary 字符串中的起止位置
        word_spans = []
        cursor = 0
        for w in words:
            # 找到单词起始位置
            start = summary_text.find(w, cursor)
            if start == -1: continue  # 容错
            end = start + len(w)
            word_spans.append({"text": w, "start": start, "end": end})
            cursor = end

        # 3. 将 BERT Subwords 映射到这些单词上
        mapped_data = []  # List of [word_text, [subword_indices]]

        # 初始化映射列表
        for span in word_spans:
            mapped_data.append([span['text'], []])

        # 遍历 Subword Offsets
        for i, (b_start, b_end) in enumerate(bert_offsets):
            b_center = (b_start + b_end) / 2

            # 查找这个 Subword 属于哪个单词
            for idx, span in enumerate(word_spans):
                if span['start'] <= b_center < span['end']:
                    mapped_data[idx][1].append(i)
                    break

        return mapped_data

    def analyze(self, code, summary):
        # ... (前 1-4 步代码完全不变) ...
        # 1. BERT 推理
        inputs = self.tokenizer(code, summary, return_tensors="pt", return_offsets_mapping=True, truncation=True)
        input_ids = inputs["input_ids"].to(self.device)
        offsets = inputs["offset_mapping"][0].cpu().numpy()

        with torch.no_grad():
            outputs = self.model(input_ids, output_attentions=True)
        raw_attn = outputs.attentions[10][0].mean(dim=0).cpu().numpy()

        # 2. 区域划分
        sep_idxs = (input_ids[0] == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
        c_range = (1, sep_idxs[0].item())
        s_range = (sep_idxs[1].item() + 1, sep_idxs[2].item() if len(sep_idxs) > 2 else len(input_ids[0]))

        # 3. 获取 AST Sequence
        splitter = StatementSplitter(code)
        splitter.visit(ast.parse(code).body)
        st_sequence = splitter.get_sequence()

        all_ast_tokens = []
        for seq in st_sequence:
            all_ast_tokens.extend(seq.tokens)

        # 4. 建立映射关系
        mapped_code = self._map_code_subwords_to_ast(offsets[c_range[0]:c_range[1]], all_ast_tokens)
        mapped_summary = self._map_summary_subwords_to_words(offsets[s_range[0]:s_range[1]], summary)

        # 5. 双向聚合矩阵
        raw_cross = raw_attn[s_range[0]:s_range[1], c_range[0]:c_range[1]]

        # 列聚合
        semi_agg_matrix = np.zeros((raw_cross.shape[0], len(mapped_code)))
        for j, (token_obj, sub_idxs) in enumerate(mapped_code):
            if sub_idxs:
                semi_agg_matrix[:, j] = raw_cross[:, sub_idxs].mean(axis=1)

        # 行聚合
        final_matrix = np.zeros((len(mapped_summary), len(mapped_code)))
        for i, (word_text, sub_idxs) in enumerate(mapped_summary):
            if sub_idxs:
                final_matrix[i, :] = semi_agg_matrix[sub_idxs, :].mean(axis=0)

        statement_stats = []
        for seq in st_sequence:
            # 找到属于该语句的 mapped_code 索引
            relevant_col_indices = []
            for j, (m_tok, _) in enumerate(mapped_code):
                # 对象同一性判断：判断映射后的 token 是否属于当前语句
                if m_tok in seq.tokens:
                    relevant_col_indices.append(j)

            # 计算分数
            if relevant_col_indices:
                # 均值：Summary 整体对该语句的平均关注度
                score = final_matrix[:, relevant_col_indices].mean()
            else:
                score = 0.0

            # 封装结果
            statement_stats.append({
                "type": seq.statement_type,
                "score": float(score),  # 确保是 float
                "tokens": [t.text for t in seq.tokens],  # 方便预览
                "preview": " ".join([t.text for t in seq.tokens])
            })

        # 返回: Summary映射, Code映射, 矩阵, 【计算好分数的语句列表】
        return mapped_summary, mapped_code, final_matrix, statement_stats


# ==========================================
# 运行与展示
# ==========================================
if __name__ == "__main__":
    code = """
def generate(ctx, url, *args, **kwargs):
    file_previews = ctx.obj['file_previews']
    options = {}
    metadata = kwargs['metadata']
    width = kwargs['width']
    height = kwargs['height']
    output_format = kwargs['format']
    if metadata:
        options['metadata'] = metadata.split(',')
    if width:
        options.setdefault('size', {})
        options['size']['width'] = width
    if height:
        options.setdefault('size', {})
        options['size']['height'] = height
    if output_format:
        options['format'] = output_format
    else:
        options['format'] = '1'
    results = file_previews.generate(url, **options)
    click.echo(results)
    """
    summary = "Generate preview for URL"

    analyzer = DeepAttentionAnalyzer()

    mapped_sum, mapped_code, matrix, st_sequence = analyzer.analyze(code, summary)

    print(f"\n{'=' * 30} VIEW 1: Natural Summary Word -> Top 3 AST Tokens {'=' * 30}")

    header_fmt = "{:<15} | {:<30} | {:<30} | {:<30}"
    print(header_fmt.format("Word", "Rank 1", "Rank 2", "Rank 3"))
    print("-" * 115)

    # 遍历 Summary 的自然单词
    for i, (word_text, sub_idxs) in enumerate(mapped_sum):
        # 只有当该单词在 BERT 中有对应的 Subwords 时才有分数
        if not sub_idxs: continue
        scores = matrix[i]  # 这是聚合后的分数 (Mean Pooling)
        # 取前3名
        top_k_indices = scores.argsort()[-3:][::-1]
        rank_strs = []
        for idx in top_k_indices:
            # 获取 Code Token 对象
            token_obj = mapped_code[idx][0]
            score = scores[idx]
            tok_text = token_obj.text
            if len(tok_text) > 15: tok_text = tok_text[:12] + "..."
            rank_strs.append(f"{tok_text} ({score:.4f})")

        while len(rank_strs) < 3: rank_strs.append("-")

        print(header_fmt.format(word_text, rank_strs[0], rank_strs[1], rank_strs[2]))

    print(f"\n{'=' * 30} VIEW 2: Statement Analysis {'=' * 30}")
    print(f"{'Type':<12} | {'Score':<8} | {'Preview'}")
    print("-" * 100)

    for seq in st_sequence:
        # --- 修正点 1: Preview 直接使用 AST 的原始 Token 列表 ---
        # 这样保证了所有标点符号一个不少
        preview = " ".join([t.text for t in seq.tokens])
        # --- 修正点 2: 计算分数时再去查找映射 ---
        relevant_col_indices = []
        for j, (m_tok, _) in enumerate(mapped_code):
            # 判断 mapped_code 里的这个 token 是否属于当前语句
            # 注意：这里对比的是对象内存地址 (is check)，因为 m_tok 就是从 seq.tokens 里来的引用
            if m_tok in seq.tokens:
                relevant_col_indices.append(j)

        if relevant_col_indices:
            score = matrix[:, relevant_col_indices].mean()
        else:
            score = 0.0

        print(f"{seq.statement_type:<12} | {score:.8f}   | {preview}")