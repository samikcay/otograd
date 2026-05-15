#include "../src/otograd.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Neuron {
    struct Tensor** w;
    struct Tensor* b;
    int w_length;
} Neuron;

typedef struct Layer {
    struct Neuron** neurons;
    int neuron_count;
} Layer;

typedef struct MLP {
    struct Layer** layers;
    int layer_count;
} MLP;

Neuron* neuron_create(int n);
Tensor* neuron_forward(Neuron* n, Tensor** x, int x_length);
Tensor** neuron_params(Neuron* n, int* param_count);
void neuron_delete(Neuron* neuron);

Layer* layer_create(int nin, int nout);
Tensor** layer_forward(Layer* l, Tensor** x, int x_length, int* out_count);
Tensor** layer_params(Layer* l, int* param_count);

MLP* mlp_create(int nin, int* layer_count, int layer_count_size);
Tensor** mlp_forward(MLP* mlp, Tensor** x, int x_length, int* out_count);
Tensor** mlp_params(MLP* mlp, int* param_count);

void print_neuron_params(Neuron* n);
int mlp_param_count(MLP* mlp);

#ifdef __cplusplus
}
#endif