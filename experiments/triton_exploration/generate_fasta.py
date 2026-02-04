import random

# 设定序列长度
sequence_lengths = [17, 28, 100, 500, 500, 2000, 2000, 5000, 5000, 10000]
mutation_rate = 0.15  # 15% 的突变比例

# 生成随机 DNA 序列
def generate_sequence(length):
    return ''.join(random.choices("ATCG", k=length))

# 生成突变序列
def mutate_sequence(seq, mutation_rate=0.15):
    seq = list(seq)  # 转换为可修改的列表
    num_mutations = int(len(seq) * mutation_rate)  # 计算突变数量
    mutation_indices = random.sample(range(len(seq)), num_mutations)  # 随机选取突变位置

    for idx in mutation_indices:
        original_base = seq[idx]
        new_base = random.choice([b for b in "ATCG" if b != original_base])  # 随机变异但不与原碱基相同
        seq[idx] = new_base

    return ''.join(seq)

# 生成 FASTA 文件
def write_fasta(ref_filename, query_filename):
    with open(ref_filename, 'w') as ref_file, open(query_filename, 'w') as query_file:
        for i, length in enumerate(sequence_lengths, start=1):
            ref_seq = generate_sequence(length)
            query_seq = mutate_sequence(ref_seq, mutation_rate)

            ref_file.write(f">{i}\n{ref_seq}\n")
            query_file.write(f">{i}\n{query_seq}\n")

# 生成 ref.fasta 和 query.fasta
write_fasta("ref.fasta", "query.fasta")

print("FASTA files 'ref.fasta' and 'query.fasta' generated successfully.")