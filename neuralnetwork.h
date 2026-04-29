/*
    Fonksiyonlar
        void init(struct s*);
    şeklinde de yazılabilirdi fakat ben
        struct* init();
    şeklinde tercih ettim
*/

#include "otograd.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Neuron Neuron;
typedef struct Layer Layer;
typedef struct MLP MLP;

Neuron* neuron_create(int n);
float neuron_forward(Neuron* n, float* x, int x_length);
Tensor** neuron_params(Neuron* n, int* param_count);
void neuron_delete(Neuron* neuron);

Layer* layer_create(int nin, int nout);
float layer_forward(Layer* l, float* x, int x_length);
Tensor** layer_params(Layer* l, int* param_count);

MLP* mlp_create(int nin, int* layer_count, int layer_count_size);
float mlp_forward(MLP* mlp, float* x, int x_length);
Tensor** mlp_params(MLP* mlp, int* param_count);

void print_neuron_params(Neuron* n);

#ifdef __cplusplus
}
#endif