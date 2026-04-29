#include "neuralnetwork.h"

struct Neuron {
    struct Tensor** w;
    struct Tensor* b;
    int w_length;
};

struct Layer {
    struct Neuron** neurons;
    int neuron_count;
};

struct MLP {
    struct Layers** layers;
    int layer_count;
};
#pragma region Neuron

Neuron* neuron_create(int nin)
{
    Neuron* neuron = ALLOC(Neuron, 1);
    
    neuron->w = DALLOC(Tensor*, nin);
    neuron->b = tensor_create(((float)rand() / (float)RAND_MAX) * 2);
    neuron->w_length = nin;

    for (int i = 0; i < nin; i++)
    {
        neuron->b = tensor_create(((float)rand() / (float)RAND_MAX) * 2); // -1 ile 1 arasi bir float değer - a float value between -1, 1
    }

    return neuron;
}

float neuron_forward(Neuron* n, float* x, int x_length)
{
    if (x == NULL || n == NULL || n->w == NULL) return 0.0f;

    assert(n->w_length == x_length);

    float wsum = n->b->data;
    for (int i = 0; i < x_length; i++)
    {
        wsum += n->w[i]->data * x[i];
    }

    return tanh(wsum);
}

Tensor** neuron_params(Neuron* n, int* param_count)
{
    if (n == NULL) return NULL;

    Tensor** tensor_list = DALLOC(Tensor*, n->w_length + 1);
    int i;
    for (i = 0; i < n->w_length; i++)
    {
        tensor_list[i] = n->w[i];
    }
    tensor_list[i] = n->b;
    *param_count = i;

    return tensor_list;
}

void neuron_delete(Neuron* neuron)
{
    free(neuron->w);
    free(neuron);
}

#pragma endregion

#pragma region Layer

Layer* layer_create(int nin, int nout)
{
    Layer* layer = ALLOC(Layer, 1);
    layer->neurons = DALLOC(Neuron*, nout);
    layer->neuron_count = nout;

    for (int i = 0; i < nin; i++)
    {
        layer->neurons[i] = ALLOC(Neuron, nin);
    }

    return layer;
}

float layer_forward(Layer* l, float* x, int x_length)
{
    float wsum = 0.0f;

    for (int i = 0; i < l->neuron_count; i++)
    {
        wsum = neuron_forward(l->neurons[i], x, x_length);
    }

    return wsum;
}

Tensor** layer_params(Layer* l, int* param_count)
{
    if (l == NULL || l->neurons == NULL || param_count == NULL) return NULL;
    int w_count = l->neurons[0]->w_length;
    Tensor** tensor_list = DALLOC(Tensor*, l->neuron_count * (w_count + 1));

    for (int i = 0; i < l->neuron_count; i++)
    {
        Neuron* n = l->neurons[i];
        int j = 0;
        for (j = 0; j < w_count; j++)
        {
            tensor_list[i * (w_count + 1) + j] = n->w[j];
        }
        tensor_list[i * (w_count + 1) + j] = n->b;
    }

    *param_count = (w_count + 1) * l->neuron_count;

    return tensor_list;
}

#pragma endregion

#pragma region MLP

MLP* mlp_create(int nin, int* layer_count, int layer_count_size)
{

}

float mlp_forward()
{

}

Tensor** mlp_params(MLP* mlp)
{
    
}

#pragma endregion

void print_neuron_params(Neuron* n)
{
    printf("(");
    for (int i = 0; i < n->w_length; i++)
    {
        printf("%f, ", n->w[i]);
    }
    printf("%f)", n->b);
}