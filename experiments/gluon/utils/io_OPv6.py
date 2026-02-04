# prototype/utils/io_OPv6.py

def read_fasta(path: str) -> list[str]:
    # This is the original function, you can keep it for other purposes
    # or replace it if it's no longer needed.
    seqs = []
    cur = ""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if cur: seqs.append(cur)
                cur = ""
            else:
                cur += line
        if cur: seqs.append(cur)
    return seqs

def read_fasta_as_bytes(path: str) -> list[bytes]:
    """
    Reads sequences from a FASTA file and yields them as encoded bytes.
    This is more efficient as it avoids creating intermediate Python strings
    and moves the encoding step out of the critical performance path.
    """
    seqs = []
    # Use a list of byte chunks to avoid slow string concatenation
    cur_chunks = []
    with open(path, 'rb') as f: # Read file in binary mode
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(b">"):
                if cur_chunks:
                    seqs.append(b"".join(cur_chunks))
                cur_chunks = []
            else:
                cur_chunks.append(line)
        if cur_chunks:
            seqs.append(b"".join(cur_chunks))
    return seqs