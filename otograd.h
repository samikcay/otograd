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
#include <assert.h>
#include <stdio.h>

#define ALLOC(x, y) ((x*)malloc(sizeof(x) * y))
#define DALLOC(x, y) ((x**)malloc(sizeof(x*) * y))

#define EULERNUM 2.71828182845904523536

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_GRAPH_SIZE 1000

typedef struct Tensor {
    float data;
    float grad;
    struct Tensor** _from;
    int _num_from;
    char _op;
    int visited;
} Tensor;

Tensor* tensor_create(float data);
Tensor* tensor_add(Tensor* t1, Tensor* t2);
Tensor* tensor_mul(Tensor* t1, Tensor* t2);
Tensor* tensor_sub(Tensor* t1, Tensor* t2);
Tensor* tensor_div(Tensor* t1, Tensor* t2);
void tensor_free(Tensor* t);
void tensor_free_all(Tensor* t);

Tensor** topological_sort(Tensor* head, int* out_count);
int build_topo(Tensor* t, Tensor** tensor_list, int* topo_count);
void backward(Tensor* head);
void tensor_backward(Tensor* t);
void reset_visited(Tensor* t);

#ifdef __cplusplus
}
#endif
