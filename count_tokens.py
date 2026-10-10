import os
import sys
import json
import time
import unicodedata

SPECIAL_TOKENS_LIST = ['<|bos|>', '<|eos|>', '<|pad|>', '<|unk|>', '<|user|>', '<|assistant|>', '<|end|>']

def load_tokenizer(directory):
    from tokenizers import Tokenizer, decoders
    from tokenizers.pre_tokenizers import Whitespace
    import torch

    bin_path = os.path.join(directory, "tokenizer_incremental.bin")
    json_path = os.path.join(directory, "tokenizer_incremental_tokenizer.json")

    if not os.path.exists(bin_path):
        print(f"错误: 未找到 {bin_path}")
        return None
    if not os.path.exists(json_path):
        print(f"错误: 未找到 {json_path}")
        return None

    data = torch.load(bin_path, map_location='cpu', weights_only=False)
    tokenizer = Tokenizer.from_file(json_path)
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = decoders.BPEDecoder()

    vocab = tokenizer.get_vocab()
    unk_id = vocab.get('<|unk|>', 3)
    user_id = vocab.get('<|user|>', 4)
    assistant_id = vocab.get('<|assistant|>', 5)
    end_id = vocab.get('<|end|>', 6)
    bos_id = vocab.get('<|bos|>', 0)
    eos_id = vocab.get('<|eos|>', 1)

    vocab_size = tokenizer.get_vocab_size()
    print(f"词表加载成功，词汇量: {vocab_size}")
    return tokenizer, unk_id, user_id, assistant_id, end_id, bos_id, eos_id, vocab_size

def count_txt_tokens(tokenizer, filepath, unk_id, vocab_size):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    non_empty = [l.rstrip('\n\r') + '\n' for l in lines if l.strip()]
    total_lines = len(non_empty)

    if not non_empty:
        return 0, 0

    batch_size = 8192
    total_tokens = 0
    for i in range(0, len(non_empty), batch_size):
        chunk = non_empty[i:i+batch_size]
        normalized = [unicodedata.normalize('NFKC', t).lower() for t in chunk]
        encodings = tokenizer.encode_batch(normalized)
        for enc in encodings:
            ids = enc.ids
            total_tokens += sum(1 if tid < vocab_size else 1 for tid in ids)

    return total_tokens, total_lines

def count_jsonl_tokens(tokenizer, filepath, unk_id, user_id, assistant_id, end_id, bos_id, eos_id, vocab_size):
    all_texts = []
    sample_meta = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            convs = data.get('conversations', [])
            if not convs:
                continue
            turn_indices = []
            for turn in convs:
                role = turn.get('role', '')
                content = turn.get('content', '')
                if role in ('user', 'assistant'):
                    idx = len(all_texts)
                    all_texts.append(content)
                    turn_indices.append((role, idx))
            if turn_indices:
                sample_meta.append(turn_indices)

    if not all_texts:
        return 0, 0, 0

    batch_size = 8192
    token_counts = [0] * len(all_texts)
    for i in range(0, len(all_texts), batch_size):
        chunk = all_texts[i:i+batch_size]
        normalized = [unicodedata.normalize('NFKC', t).lower() for t in chunk]
        encodings = tokenizer.encode_batch(normalized)
        for j, enc in enumerate(encodings):
            token_counts[i+j] = len(enc.ids)

    total_tokens = 0
    total_turns = 0
    for turn_indices in sample_meta:
        sample_tokens = 1  # bos
        for role, idx in turn_indices:
            sample_tokens += 1  # role token
            sample_tokens += token_counts[idx]
            sample_tokens += 1  # end token
            total_turns += 1
        sample_tokens += 1  # eos
        total_tokens += sample_tokens

    return total_tokens, total_turns, len(sample_meta)

def scan_files(directory):
    files = []
    for f in sorted(os.listdir(directory)):
        if f.startswith('.'):
            continue
        if f.endswith('.txt') or f.endswith('.jsonl'):
            try:
                size = os.path.getsize(os.path.join(directory, f))
            except OSError:
                size = 0
            files.append((f, size))
    return files

def format_size(size):
    if size >= 1024**3:
        return f"{size / 1024**3:.1f} GB"
    elif size >= 1024**2:
        return f"{size / 1024**2:.1f} MB"
    elif size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"

def format_tokens(tokens):
    if tokens >= 1_000_000:
        return f"{tokens:,} ({tokens / 1_000_000:.2f}M)"
    elif tokens >= 1_000:
        return f"{tokens:,} ({tokens / 1_000:.1f}K)"
    return f"{tokens:,}"

def estimate_steps(tokens, seq_len=4096, batch_size=1, accum=1):
    return max(1, tokens // (seq_len * batch_size * accum))

def main():
    directory = os.path.dirname(os.path.abspath(__file__))
    print(f"\n=== Token 计数器 ===")
    print(f"工作目录: {directory}\n")

    result = load_tokenizer(directory)
    if result is None:
        input("按回车退出...")
        return
    tokenizer, unk_id, user_id, assistant_id, end_id, bos_id, eos_id, vocab_size = result

    while True:
        files = scan_files(directory)
        if not files:
            print("当前目录下未找到 .txt 或 .jsonl 文件。")
            input("按回车退出...")
            return

        print(f"\n{'='*60}")
        print("可用文件：")
        print(f"{'='*60}")
        for i, (name, size) in enumerate(files, 1):
            ext = os.path.splitext(name)[1]
            tag = "[txt]" if ext == '.txt' else "[jsonl]"
            print(f"  {i:2d}. {tag:7s} {name} ({format_size(size)})")
        print(f"  {len(files)+1:2d}. [全部] 扫描所有文件")
        print(f"   0. 退出")
        print(f"{'='*60}")

        raw = input("\n请输入文件序号: ").strip()
        try:
            choice = int(raw)
        except ValueError:
            print("输入无效，请输入数字。")
            continue

        if choice == 0:
            break

        if choice == len(files) + 1:
            total_all_tokens = 0
            total_all_turns = 0
            total_all_samples = 0
            print(f"\n{'='*60}")
            print("扫描所有文件...")
            print(f"{'='*60}")
            start_time = time.time()
            for name, size in files:
                filepath = os.path.join(directory, name)
                ext = os.path.splitext(name)[1].lower()
                t0 = time.time()
                if ext == '.txt':
                    tokens, lines = count_txt_tokens(tokenizer, filepath, unk_id, vocab_size)
                    total_all_tokens += tokens
                    dt = time.time() - t0
                    print(f"  {name:40s} {format_tokens(tokens):>20s} tokens  ({lines:,} 行)  [{dt:.1f}s]")
                elif ext == '.jsonl':
                    tokens, turns, samples = count_jsonl_tokens(tokenizer, filepath, unk_id,
                                                                 user_id, assistant_id, end_id, bos_id, eos_id, vocab_size)
                    total_all_tokens += tokens
                    total_all_turns += turns
                    total_all_samples += samples
                    dt = time.time() - t0
                    print(f"  {name:40s} {format_tokens(tokens):>20s} tokens  ({samples:,} 样本, {turns:,} 轮)  [{dt:.1f}s]")
            elapsed = time.time() - start_time
            print(f"{'='*60}")
            print(f"  合计: {format_tokens(total_all_tokens)} tokens")
            if total_all_samples > 0:
                print(f"  JSONL: {total_all_samples:,} 样本, {total_all_turns:,} 轮")
            print(f"  耗时: {elapsed:.1f}s")
            est = estimate_steps(total_all_tokens)
            print(f"  预估训练步数 (seq=4096, batch=1): {est:,} 步")
            print(f"{'='*60}")
            continue

        if choice < 1 or choice > len(files):
            print(f"序号超出范围，请输入 0-{len(files)+1}。")
            continue

        name, size = files[choice - 1]
        filepath = os.path.join(directory, name)
        ext = os.path.splitext(name)[1].lower()

        print(f"\n正在统计 {name} ({format_size(size)})...")
        start_time = time.time()

        if ext == '.txt':
            tokens, lines = count_txt_tokens(tokenizer, filepath, unk_id, vocab_size)
            elapsed = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"  文件: {name}")
            print(f"  类型: 纯文本 (txt)")
            print(f"  大小: {format_size(size)}")
            print(f"  行数: {lines:,}")
            print(f"  Token数: {format_tokens(tokens)}")
            if lines > 0:
                print(f"  平均每行: {tokens / lines:.1f} tokens")
            print(f"  耗时: {elapsed:.1f}s")
            est = estimate_steps(tokens)
            print(f"  预估训练步数 (seq=4096, batch=1): {est:,} 步")
            print(f"{'='*60}")

        elif ext == '.jsonl':
            tokens, turns, samples = count_jsonl_tokens(tokenizer, filepath, unk_id,
                                                         user_id, assistant_id, end_id, bos_id, eos_id, vocab_size)
            elapsed = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"  文件: {name}")
            print(f"  类型: 对话数据 (jsonl)")
            print(f"  大小: {format_size(size)}")
            print(f"  样本数: {samples:,}")
            print(f"  对话轮数: {turns:,}")
            print(f"  Token数: {format_tokens(tokens)}")
            if samples > 0:
                print(f"  平均每样本: {tokens / samples:.1f} tokens")
                print(f"  平均每轮: {tokens / turns:.1f} tokens")
            print(f"  耗时: {elapsed:.1f}s")
            est = estimate_steps(tokens)
            print(f"  预估训练步数 (seq=4096, batch=1): {est:,} 步")
            print(f"{'='*60}")

if __name__ == "__main__":
    main()
