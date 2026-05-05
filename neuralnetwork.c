#include "neuralnetwork.h"

#pragma region Neuron

Neuron* neuron_create(int nin)
{
	Neuron* neuron = ALLOC(Neuron, 1);

	neuron->w = DALLOC(Tensor, nin);
	neuron->b = tensor_create(((float)rand() / (float)RAND_MAX) * 2);
	neuron->w_length = nin;

	for (int i = 0; i < nin; i++)
	{
		neuron->w[i] = tensor_create(((float)rand() / (float)RAND_MAX) * 2); // -1 ile 1 arasi bir float değer - a float value between -1, 1
	}

	return neuron;
}

float neuron_forward(Neuron* n, float* xs, int xs_length)
{
	if (xs == NULL || n == NULL || n->w == NULL) return 0.0f;

	assert(n->w_length == xs_length);

	float wsum = n->b->data;
	for (int i = 0; i < xs_length; i++)
	{
		wsum += n->w[i]->data * xs[i];
	}

	return tanh(wsum);
}

Tensor** neuron_params(Neuron* n, int* param_count)
{
	assert(n != NULL && n->w != NULL && n->b != NULL);

	Tensor** tensor_list = DALLOC(Tensor, n->w_length + 1);
	int i;
	for (i = 0; i < n->w_length; i++)
	{
		tensor_list[i] = n->w[i];
	}
	tensor_list[i] = n->b;
	*param_count = n->w_length + 1;

	return tensor_list;
}

void neuron_delete(Neuron* neuron)
{
	if (neuron == NULL) return;

	for (int i = 0; i < neuron->w_length; i++)
	{
		tensor_free(neuron->w[i]);
	}
	tensor_free(neuron->b);
	free(neuron->w);
	free(neuron);
}

#pragma endregion

#pragma region Layer

Layer* layer_create(int nin, int nout)
{
	Layer* layer = ALLOC(Layer, 1);
	layer->neurons = DALLOC(Neuron, nout);
	layer->neuron_count = nout;

	for (int i = 0; i < nout; i++)
	{
		layer->neurons[i] = neuron_create(nin);
	}

	return layer;
}

float layer_forward(Layer* l, float* xs, int xs_length)
{
	assert(l != NULL && l->neurons != NULL && xs != NULL);
	float wsum = 0.0f;

	for (int i = 0; i < l->neuron_count; i++)
	{
		wsum += neuron_forward(l->neurons[i], xs, xs_length);
	}

	return wsum;
}

Tensor** layer_params(Layer* l, int* param_count)
{
	assert(l != NULL && l->neurons != NULL);
	int w_count = l->neurons[0]->w_length;
	Tensor** tensor_list = DALLOC(Tensor, l->neuron_count * (w_count + 1));

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
	MLP* mlp = ALLOC(MLP, 1);
	mlp->layers = DALLOC(Layer, layer_count_size);
	mlp->layer_count = layer_count_size;

	for (int i = 0; i < layer_count_size; i++)
	{
		int n_in = (i == 0) ? nin : layer_count[i - 1];
		int n_out = layer_count[i];
		mlp->layers[i] = layer_create(n_in, n_out);
	}

	return mlp;
}

float mlp_forward(MLP* mlp, float* xs, int xs_count)
{
	assert(mlp != NULL && mlp->layers != NULL && xs != NULL);

	float* current_xs = xs;
	int current_count = xs_count;
	float* previous_outputs = NULL;
	float* outputs = NULL;
	int output_count = 0;

	for (int i = 0; i < mlp->layer_count; i++)
	{
		Layer* layer = mlp->layers[i];
		output_count = layer->neuron_count;
		outputs = ALLOC(float, output_count);

		for (int j = 0; j < output_count; j++)
		{
			outputs[j] = neuron_forward(layer->neurons[j], current_xs, current_count);
		}

		free(previous_outputs);
		previous_outputs = outputs;
		current_xs = outputs;
		current_count = output_count;
	}

	float result = 0.0f;
	for (int i = 0; i < output_count; i++)
	{
		result += outputs[i];
	}

	free(outputs);
	return result;
}

Tensor** mlp_params(MLP* mlp, int* param_count)
{
	assert(mlp != NULL && mlp->layers != NULL);
	*param_count = mlp_param_count(mlp);
	Tensor** tensor_list = DALLOC(Tensor, *param_count);

	int index = 0;
	for (int i = 0; i < mlp->layer_count; i++)
	{
		for (int j = 0; j < mlp->layers[i]->neuron_count; j++)
		{
			Neuron* n = mlp->layers[i]->neurons[j];
			for (int k = 0; k < n->w_length; k++)
			{
				tensor_list[index++] = n->w[k];
			}
			tensor_list[index++] = n->b;
		}
	}
		
	return tensor_list;
}

int mlp_param_count(MLP* mlp)
{
	assert(mlp != NULL && mlp->layers != NULL);

	int count = 0;
	for (int i = 0; i < mlp->layer_count; i++)
	{
		for (int j = 0; j < mlp->layers[i]->neuron_count; j++)
		{
			count += mlp->layers[i]->neurons[j]->w_length + 1; // +1 for bias
		}
	}

	return count;
}

#pragma endregion

void print_neuron_params(Neuron* n)
{
	printf("(");
	for (int i = 0; i < n->w_length; i++)
	{
		printf("%f, ", n->w[i]->data);
	}
	printf("%f)", n->b->data);
}

float ReLU(float x)
{
	return (x > 0) ? x : 0;
}
