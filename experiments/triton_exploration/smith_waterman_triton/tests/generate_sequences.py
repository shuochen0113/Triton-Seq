from tests.sw_benchmark_interquery import generate_pairs, write_fasta, AGATHA_ROOT

q_list, r_list = generate_pairs(10000)
DATASET_DIR = AGATHA_ROOT / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)
write_fasta(q_list, DATASET_DIR / "generated_query.fasta")
write_fasta(r_list, DATASET_DIR / "generated_ref.fasta")