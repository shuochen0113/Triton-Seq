#include "gasal.h"

#include "args_parser.h"

#include "res.h"


gasal_res_t *gasal_res_new_host(uint32_t max_n_alns, Parameters *params)
{
	cudaError_t err;
	gasal_res_t *res = NULL;


	res = (gasal_res_t *)malloc(sizeof(gasal_res_t));

	CHECKCUDAERROR(cudaHostAlloc(&(res->aln_score), max_n_alns * sizeof(int32_t),cudaHostAllocDefault));
	
	
	if(res ==NULL)
	{
		fprintf(stderr,  "Malloc error on res host ");
		exit(1);
	}

	CHECKCUDAERROR(cudaHostAlloc(&(res->query_batch_end),max_n_alns * sizeof(uint32_t),cudaHostAllocDefault));
	CHECKCUDAERROR(cudaHostAlloc(&(res->target_batch_end),max_n_alns * sizeof(uint32_t),cudaHostAllocDefault));
	res->query_batch_start = NULL;
	res->target_batch_start = NULL;

    int max_seq_len = params->kernel_align_num; // Or use gpu_storage->maximum_sequence_length if available
    size_t traceback_size = max_n_alns * max_seq_len * max_seq_len * sizeof(uint8_t);
    res->traceback = (uint8_t *)malloc(traceback_size);
    
    // Allocate buffers for the aligned sequences.
    res->aligned_query = (char *)malloc(max_n_alns * (max_seq_len + 1) * sizeof(char));
    res->aligned_target = (char *)malloc(max_n_alns * (max_seq_len + 1) * sizeof(char));
    
    return res;
}



gasal_res_t *gasal_res_new_device(gasal_res_t *device_cpy)
{
	cudaError_t err;


	
    // create class storage on device and copy top level class
    gasal_res_t *d_c;
    CHECKCUDAERROR(cudaMalloc((void **)&d_c, sizeof(gasal_res_t)));
	//    CHECKCUDAERROR(cudaMemcpy(d_c, res, sizeof(gasal_res_t), cudaMemcpyHostToDevice));



    // copy pointer to allocated device storage to device class
    CHECKCUDAERROR(cudaMemcpy(&(d_c->aln_score), &(device_cpy->aln_score), sizeof(int32_t*), cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpy(&(d_c->query_batch_start), &(device_cpy->query_batch_start), sizeof(int32_t*), cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpy(&(d_c->target_batch_start), &(device_cpy->target_batch_start), sizeof(int32_t*), cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpy(&(d_c->query_batch_end), &(device_cpy->query_batch_end), sizeof(int32_t*), cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpy(&(d_c->target_batch_end), &(device_cpy->target_batch_end), sizeof(int32_t*), cudaMemcpyHostToDevice));





	return d_c;
}




gasal_res_t *gasal_res_new_device_cpy(uint32_t max_n_alns, Parameters *params)
{
	cudaError_t err;
    gasal_res_t *res = (gasal_res_t *)malloc(sizeof(gasal_res_t));
    CHECKCUDAERROR(cudaMalloc(&(res->aln_score), max_n_alns * sizeof(int32_t)));
    CHECKCUDAERROR(cudaMalloc(&(res->query_batch_end), max_n_alns * sizeof(uint32_t)));
    CHECKCUDAERROR(cudaMalloc(&(res->target_batch_end), max_n_alns * sizeof(uint32_t)));
    // Allocate device traceback: assume max_seq_len from parameters.
    int max_seq_len = params->kernel_align_num; // adjust as needed
    size_t traceback_size = max_n_alns * max_seq_len * max_seq_len * sizeof(uint8_t);
    CHECKCUDAERROR(cudaMalloc(&(res->traceback), traceback_size));
    // Optionally allocate device aligned_query and aligned_target if needed.
    res->query_batch_start = NULL;
    res->target_batch_start = NULL;
    return res;
}


// TODO : make 2 destroys for host and device
void gasal_res_destroy_host(gasal_res_t *res) 
{
    cudaError_t err;
    if (res == NULL)
        return;

    if (res->aln_score != NULL) CHECKCUDAERROR(cudaFreeHost(res->aln_score));
    if (res->query_batch_start != NULL) CHECKCUDAERROR(cudaFreeHost(res->query_batch_start));
    if (res->target_batch_start != NULL) CHECKCUDAERROR(cudaFreeHost(res->target_batch_start));
    if (res->query_batch_end != NULL) CHECKCUDAERROR(cudaFreeHost(res->query_batch_end));
    if (res->target_batch_end != NULL) CHECKCUDAERROR(cudaFreeHost(res->target_batch_end));

    // ✅ Free traceback storage
    if (res->traceback != NULL) free(res->traceback);
    if (res->aligned_query != NULL) free(res->aligned_query);
    if (res->aligned_target != NULL) free(res->aligned_target);

    free(res);
}

void gasal_res_destroy_device(gasal_res_t *device_res, gasal_res_t *device_cpy) 
{
    cudaError_t err;
    if (device_cpy == NULL || device_res == NULL)
        return;

    if (device_cpy->aln_score != NULL) CHECKCUDAERROR(cudaFree(device_cpy->aln_score));
    if (device_cpy->query_batch_start != NULL) CHECKCUDAERROR(cudaFree(device_cpy->query_batch_start));
    if (device_cpy->target_batch_start != NULL) CHECKCUDAERROR(cudaFree(device_cpy->target_batch_start));
    if (device_cpy->query_batch_end != NULL) CHECKCUDAERROR(cudaFree(device_cpy->query_batch_end));
    if (device_cpy->target_batch_end != NULL) CHECKCUDAERROR(cudaFree(device_cpy->target_batch_end));

    // ✅ Free traceback storage
    if (device_cpy->traceback != NULL) CHECKCUDAERROR(cudaFree(device_cpy->traceback));
    if (device_cpy->aligned_query != NULL) CHECKCUDAERROR(cudaFree(device_cpy->aligned_query));
    if (device_cpy->aligned_target != NULL) CHECKCUDAERROR(cudaFree(device_cpy->aligned_target));

    CHECKCUDAERROR(cudaFree(device_res));
    free(device_cpy);
}

void reconstruct_sequence(uint8_t *traceback, char *aligned_query, char *aligned_target, int query_len, int ref_len) {
    int i = ref_len - 1;
    int j = query_len - 1;
    int pos = 0;
    // Temporary buffers to hold the alignment in reverse order.
    char temp_query[query_len + ref_len + 1];
    char temp_target[query_len + ref_len + 1];
    while(i >= 0 && j >= 0) {
        uint8_t move = traceback[i * query_len + j];
        if (move == 0) { // diagonal
            // Here, you need to fetch the corresponding base from the original sequences.
            // For simplicity, assume you have functions get_query_base(i, j) and get_ref_base(i, j).
            temp_query[pos] = get_query_base(j); // you must implement or pass original sequences
            temp_target[pos] = get_ref_base(i);
            i--; j--;
        } else if (move == 1) { // left
            temp_query[pos] = get_query_base(j);
            temp_target[pos] = '-';
            j--;
        } else if (move == 2) { // up
            temp_query[pos] = '-';
            temp_target[pos] = get_ref_base(i);
            i--;
        }
        pos++;
    }
    // Reverse the strings
    for (int k = 0; k < pos; k++) {
        aligned_query[k] = temp_query[pos - k - 1];
        aligned_target[k] = temp_target[pos - k - 1];
    }
    aligned_query[pos] = '\0';
    aligned_target[pos] = '\0';
}
