import ast
import tokenize
from io import BytesIO


# ==========================================
# 1. 基础数据结构
# ==========================================

class CodeToken:
    """
    最小的代码单元，带有精准的坐标信息。
    """

    def __init__(self, text, start_idx, end_idx, type_label="Token"):
        self.text = text
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.type = type_label

    def __repr__(self):
        return f"'{self.text}'"


class SequenceNode:
    """
    语句节点，包含该语句下的所有 Token (有序列表)
    """

    def __init__(self, statement_type, tokens):
        self.statement_type = statement_type
        self.tokens = tokens  # List[CodeToken]

        # 计算语句整体范围
        if tokens:
            self.start_char = tokens[0].start_idx
            self.end_char = tokens[-1].end_idx
        else:
            self.start_char = -1
            self.end_char = -1

    def __repr__(self):
        # 方便调试查看
        return f"<{self.statement_type}>: {' '.join([t.text for t in self.tokens])}"


# ==========================================
# 2. 精准 Tokenizer (基于 Python 标准库)
# ==========================================

class PreciseTokenizer:
    def __init__(self, source_code):
        self.source_code = source_code
        self.tokens = []
        self._tokenize()

    def _tokenize(self):
        # tokenize 库需要 bytes 输入
        try:
            tokens = list(tokenize.tokenize(BytesIO(self.source_code.encode('utf-8')).readline))
        except tokenize.TokenError:
            print("Warning: Tokenization failed (possibly incomplete code).")
            return

        # 预计算每一行的起始绝对偏移量，用于将 (row, col) 转为 abs_index
        lines = self.source_code.split('\n')
        line_offsets = [0]
        for line in lines:
            line_offsets.append(line_offsets[-1] + len(line) + 1)  # +1 是换行符

        for tok in tokens:
            # 过滤掉非代码内容 (编码声明, 结束符, 纯换行, 注释)
            if tok.type in [tokenize.ENCODING, tokenize.ENDMARKER, tokenize.NL, tokenize.COMMENT, tokenize.INDENT,
                            tokenize.DEDENT]:
                continue
            # 有些 NEWLINE 是语句结束的标志，但在 AST 映射中我们通常不需要它作为 Token
            if tok.type == tokenize.NEWLINE:
                continue

            # 获取位置信息
            s_row, s_col = tok.start
            e_row, e_col = tok.end

            # 转换为绝对坐标 (tokenize 行号从1开始)
            if s_row > len(line_offsets): continue
            start_abs = line_offsets[s_row - 1] + s_col
            end_abs = line_offsets[e_row - 1] + e_col

            self.tokens.append(CodeToken(tok.string, start_abs, end_abs, str(tok.type)))


# ==========================================
# 3. 核心类：StatementSplitter
# ==========================================

class StatementSplitter:
    def __init__(self, source_code):
        self.source_code = source_code
        self.tokenizer = PreciseTokenizer(source_code)
        self.all_tokens = self.tokenizer.tokens
        self.sequence = []  # List[SequenceNode]

    def visit(self, node, override_type=None):
        if isinstance(node, list):
            for item in node: self.visit(item)
            return
        if not isinstance(node, ast.AST): return

        # 1. 计算 AST 节点的字符范围
        start_char, end_char = self._get_node_range(node)

        if start_char != -1:
            # 2. 提取该范围内的所有 Token
            node_tokens = self._filter_tokens(node, start_char, end_char)

            # 3. --- 核心修改：智能类型推断 ---
            stmt_type = override_type if override_type else self._refine_type(node)

            # 只有包含 Token 的语句才加入序列
            if node_tokens:
                self.sequence.append(SequenceNode(stmt_type, node_tokens))

        # 4. 递归处理子节点
        self._process_body(node)

    def _refine_type(self, node):
        """
        根据节点内部结构，细化语句类型。
        主要是将通用的 Expr 和 Assign 细化为 MethodCall 等。
        """
        base_type = node.__class__.__name__

        # 情况 A: 独立的表达式语句 (e.g., click.echo(results))
        if isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Call):
                # 检查是 obj.method() 还是 func()
                if isinstance(node.value.func, ast.Attribute):
                    return "MethodCall"  # 方法调用
                else:
                    return "FunctionCall"  # 普通函数调用

            # 也可以识别 await 调用
            if isinstance(node.value, ast.Await):
                return "AwaitCall"

        # 情况 B: 赋值语句中包含调用 (e.g., results = file_previews.generate(...))
        # 这一步是可选的，如果你希望保留 Assign 不变，可以注释掉下面这段
        if isinstance(node, ast.Assign):
            # 检查赋值的右侧是否是调用
            if isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Attribute):
                    return "AssignMethod"  # 赋值 + 方法调用
                else:
                    return "AssignFunc"  # 赋值 + 函数调用

        return base_type

    def _get_node_range(self, node):
        """利用 Python 3.8+ 的 get_source_segment 获取节点精确范围"""
        if not hasattr(node, 'lineno'): return -1, -1
        segment = ast.get_source_segment(self.source_code, node)
        if not segment: return -1, -1

        lines = self.source_code.split('\n')
        prefix = sum(len(line) + 1 for line in lines[:node.lineno - 1])
        start_abs = prefix + node.col_offset
        end_abs = start_abs + len(segment)
        return start_abs, end_abs

    def _filter_tokens(self, node, start, end):
        candidates = [t for t in self.all_tokens if t.start_idx >= start and t.end_idx <= end]

        # Header 截断逻辑 (包含定义和控制流)
        truncation_types = (
            ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
            ast.If, ast.For, ast.AsyncFor, ast.While,
            ast.With, ast.AsyncWith, ast.ExceptHandler
        )

        if isinstance(node, truncation_types):
            header_tokens = []
            for t in candidates:
                header_tokens.append(t)
                if t.text == ':':  # 遇到冒号停止
                    break
            return header_tokens

        return candidates

    def _process_body(self, node):
        if hasattr(node, 'body'):
            self.visit(node.body)

        # 处理 Else / Elif
        if hasattr(node, 'orelse') and node.orelse:
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                self.visit(node.orelse[0], override_type="Elif")
            else:
                self._handle_else_keyword(node)
                self.visit(node.orelse)

        if hasattr(node, 'handlers'):
            self.visit(node.handlers)

        # 处理 Finally
        if hasattr(node, 'finalbody') and node.finalbody:
            self._handle_keyword_gap(node.body + node.handlers, node.finalbody, "finally")
            self.visit(node.finalbody)

    def _handle_keyword_gap(self, prev_blocks, next_block, keyword):
        if not prev_blocks or not next_block: return

        # 寻找前一块的结束和后一块的开始
        prev_end = -1
        # 倒序查找有位置信息的节点
        for node in reversed(prev_blocks):
            _, end = self._get_node_range(node)
            if end != -1:
                prev_end = end
                break

        next_start, _ = self._get_node_range(next_block[0])

        if prev_end == -1 or next_start == -1: return

        gap_tokens = []
        found_kw = False

        for t in self.all_tokens:
            if t.start_idx >= prev_end and t.end_idx <= next_start:
                if t.text == keyword:
                    found_kw = True
                    gap_tokens.append(t)
                elif t.text == ':' and found_kw:
                    gap_tokens.append(t)
                    self.sequence.append(SequenceNode(keyword.capitalize(), gap_tokens))
                    return

    def _handle_else_keyword(self, parent_node):
        if not parent_node.body or not parent_node.orelse: return

        last_stmt = parent_node.body[-1]
        _, end_char_prev = self._get_node_range(last_stmt)

        first_else_stmt = parent_node.orelse[0]
        start_char_next, _ = self._get_node_range(first_else_stmt)

        if end_char_prev == -1 or start_char_next == -1: return

        else_tokens = []
        found_else = False

        for t in self.all_tokens:
            if t.start_idx >= end_char_prev and t.end_idx <= start_char_next:
                if t.text == 'else':
                    found_else = True
                    else_tokens.append(t)
                elif t.text == ':' and found_else:
                    else_tokens.append(t)
                    self.sequence.append(SequenceNode("Else", else_tokens))
                    return

    def get_sequence(self):
        return self.sequence


# ==========================================
# 测试代码
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
    while true:
        pass
    click.echo(results)
    """

    splitter = StatementSplitter(code)
    splitter.visit(ast.parse(code).body)

    print(f"{'Type':<12} | {'Tokens (Text)':<40} | {'Range'}")
    print("-" * 80)
    for seq in splitter.get_sequence():
        token_texts = [t.text for t in seq.tokens]
        print(f"{seq.statement_type:<12} | {str(token_texts):<40} | {seq.start_char}-{seq.end_char}")