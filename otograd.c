#include "otograd.h"

struct Tensor {
    float data;
    float grad;
    struct Tensor** _from;
    int _num_from;
    char _op;
    int visited;
};

Tensor* tensor_create(float data)
{
    Tensor* t = ALLOC(Tensor, 1);
    t->data = data;
    t->grad = 0.0f;
    t->_from = NULL;
    t->_num_from = 0;
    t->_op = ' ';
    t->visited = 0;

    return t;
}

Tensor* tensor_add(Tensor* t1, Tensor* t2)
{
    Tensor* t = ALLOC(Tensor, 1);
    t->data = t1->data + t2->data;
    t->grad = 0.0f;
    t->_op = '+';
    t->_num_from = 2;
    t->visited = 0;
    
    Tensor** from = (Tensor**)malloc(sizeof(Tensor*) * 2);
    from[0] = t1;
    from[1] = t2;
    t->_from = from;
    
    return t;
}

Tensor* tensor_mul(Tensor* t1, Tensor* t2)
{
    Tensor* t = ALLOC(Tensor, 1);
    t->data = t1->data * t2->data;
    t->grad = 0.0f;
    t->_op = '*';
    t->_num_from = 2;
    t->visited = 0;

    Tensor** from = (Tensor**)malloc(sizeof(Tensor*) * 2);
    from[0] = t1;
    from[1] = t2;
    t->_from = from;

    return t;
}

Tensor* tensor_sub(Tensor* t1, Tensor* t2)
{
    Tensor* t = ALLOC(Tensor, 1);
    t->data = t1->data - t2->data;
    t->grad = 0.0f;
    t->_op = '-';
    t->_num_from = 2;
    t->visited = 0;
    
    Tensor** from = DALLOC(Tensor*, 2);
    from[0] = t1;
    from[1] = t2;
    t->_from = from;
    
    return t;
}

Tensor* tensor_div(Tensor* t1, Tensor* t2)
{
    Tensor* t = ALLOC(Tensor, 1);
    t->data = t1->data / t2->data;
    t->grad = 0.0f;
    t->_op = '/';
    t->_num_from = 2;
    t->visited = 0;

    Tensor** from = DALLOC(Tensor*, 2);
    from[0] = t1;
    from[1] = t2;
    t->_from = from;

    return t;
}

void tensor_free(Tensor* t)
{
    if (t != NULL)
        free(t);
}

void tensor_free_all(Tensor* t) 
{
    if (t == NULL || t->visited == 1) {
        return;
    }
    
    t->visited = 1; 

    if (t->_from != NULL) {
        for (int i = 0; i < t->_num_from; i++) {
            tensor_free_all(t->_from[i]);
        }
        free(t->_from);
    }
    
    free(t);
}

Tensor** topological_sort(Tensor* head, int* out_count)
{
    Tensor** tensor_list = DALLOC(Tensor*, MAX_GRAPH_SIZE);
    int topo_count = 0;

    build_topo(head, tensor_list, &topo_count);
    
    *out_count = topo_count; 
    
    return tensor_list;
}

void build_topo(Tensor* t, Tensor** tensor_list, int* topo_count)
{
    if (t == NULL || t->visited == 1)
    {
        return;
    }

    t->visited = 1;

    if (t->_from != NULL)
    {
        for (int i = 0; i < t->_num_from; i++)
        {
            build_topo(t->_from[i], tensor_list, topo_count);
        }
    }

    tensor_list[*topo_count] = t;
    (*topo_count)++;
}

void backward(Tensor* head)
{
    int tensorCount = 0;
    Tensor** tensor_list = topological_sort(head, &tensorCount);
    head->grad = 1.0f;

    for (int i = tensorCount - 1; i >= 0; i--)
    {
        tensor_backward(tensor_list[i]);
    }
    
    free(tensor_list);
}

void inline tensor_backward(Tensor* t)
{
    if (t->_num_from > 0 && t->_from != NULL)
    {
        Tensor* t1 = t->_from[0]; Tensor* t2 = t->_from[1];
        switch (t->_op)
        {
        case '+':
            {
                t1->grad += 1.0f * t->grad;
                t2->grad += 1.0f * t->grad;
            }
            break;
        case '-':
            {
                t1->grad += 1.0f * t->grad;
                t2->grad += -1.0f * t->grad;
            }
            break;
        case '*':
            {
                t1->grad += t2->data * t->grad;
                t2->grad += t1->data * t->grad;
            }
            break;
        case '/':
            {
                t1->grad += (1.0f / t2->data) * t->grad;
                t2->grad += (-t1->data / (t2->data * t2->data)) * t->grad;
            }
            break;
        }
    }
}