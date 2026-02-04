#!/bin/bash
# Main directory (adjust the path as needed)
MAIN_DIR="/root/autodl-tmp/Cornell_intern/triton_experiment/AGAThA/"

# Directory where the test program is located
PROG_DIR="${MAIN_DIR}AGAThA/test_prog/"
# Directory for the raw, final, and score output files
OUTPUT_DIR="${MAIN_DIR}output/"
# Directory where the input dataset (FASTA files) is located
DATASET_DIR="${MAIN_DIR}dataset/"
FINAL_DIR="$PWD"

# Output files
RAW_FILE="${OUTPUT_DIR}raw.log"       # Stores all kernel execution times of all iterations        
FINAL_FILE="${OUTPUT_DIR}time.json"   # Stores the average kernel execution time per iteration
SCORE_FILE="${OUTPUT_DIR}score.log"   # Stores the alignment scores

ITER=1            # Number of iterations to run
IDLE=5            # Sleep time (in seconds) between iterations
DATASET_NAME="test"  # Name for the current dataset (will be shown in FINAL_FILE)
PROCESS="AGAThA"  # Process name (will be shown in FINAL_FILE)

# Parse command-line options
while getopts "i:" opt; do
    case "$opt" in
        i ) ITER="$OPTARG" ;;
    esac
done

mkdir -p "$OUTPUT_DIR"  # Create output directory if it does not exist

echo ">>> Running $PROCESS for $ITER iterations."

# Remove existing output files if they exist
[ -f "$RAW_FILE" ] && rm "$RAW_FILE"
[ -f "$SCORE_FILE" ] && rm "$SCORE_FILE"
[ -f "$FINAL_FILE" ] && rm "$FINAL_FILE"

iter=0
while [ "$iter" -lt "$ITER" ]; do  
    echo ">> Iteration $(($iter+1))"
    # "${PROG_DIR}manual" -p -m 3 -x 2 -q 1 -r 1 -s 3 -z 400 -w 150000 "${DATASET_DIR}ref.fasta" "${DATASET_DIR}query.fasta" "${RAW_FILE}" > "${SCORE_FILE}"
    # "${PROG_DIR}manual" -p -m 1 -x 4 -q 6 -r 2 -s 3 -z 400 -w 751 "${DATASET_DIR}generated_ref.fasta" "${DATASET_DIR}generated_query.fasta" "${RAW_FILE}" > "${SCORE_FILE}"
    ${PROG_DIR}manual -p -m 1 -x 4 -q 6 -r 2 -s 3 -z 400 -w 751 ${DATASET_DIR}ref.fasta ${DATASET_DIR}query.fasta ${RAW_FILE} > ${SCORE_FILE}
    ((iter++))
    sleep "${IDLE}s"
done

echo "$PROCESS complete."
echo "Creating output files..."

python3 /root/autodl-tmp/Cornell_intern/triton_experiment/AGAThA/misc/avg_time.py "$PROCESS" "$DATASET_NAME" "${RAW_FILE}" "${FINAL_FILE}" "$ITER" 

echo "Complete."