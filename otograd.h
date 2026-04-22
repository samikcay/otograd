// Batuhan Sami Akçay
//
// Bu kodları Andrej Karpathy'nin "The spelled-out intro to neural 
// networks and backpropagation: building micrograd" videosundan ve
// "micrograd" projesinden esinlenerek, kendim C dilinde yazmaya karar verdim. 
//
// Kodları tamamen kendim yazdım, bu sebeple olan ya da 
// olabilecek herhangi bir hata benden kaynaklıdır.
#pragma once

#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_GRAPH_SIZE 1000

typedef struct Tensor Tensor;
#define TENSOR_ALLOC(x) ((Tensor*)malloc(sizeof(Tensor) * x))
#define DTENSOR_ALLOC(x) ((Tensor**)malloc(sizeof(Tensor *) * x))


Tensor* tensor_create(float data);
Tensor* tensor_add(Tensor* t1, Tensor* t2);
Tensor* tensor_mul(Tensor* t1, Tensor* t2);
Tensor* tensor_sub(Tensor* t1, Tensor* t2);
Tensor* tensor_div(Tensor* t1, Tensor* t2);
void tensor_free(Tensor* t);
void tensor_free_all(Tensor* t);
Tensor** topological_sort(Tensor* head, int* out_count);
void build_topo(Tensor* t, Tensor** tensor_list, int* topo_count);
void backward(Tensor* head);
void inline tensor_backward(Tensor* t);

#ifdef __cplusplus
}
#endif