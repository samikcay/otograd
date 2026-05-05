#include "otograd.h"

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
    
    Tensor** from = DALLOC(Tensor, 2);
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

    Tensor** from = DALLOC(Tensor, 2);
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
    
    Tensor** from = DALLOC(Tensor, 2);
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

    Tensor** from = DALLOC(Tensor, 2);
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
    if (t == NULL) {
        return;
    }

    int tensor_count = 0;
    Tensor** tensor_list = topological_sort(t, &tensor_count);
    if (tensor_list == NULL) {
        return;
    }

    for (int i = 0; i < tensor_count; i++) {
        free(tensor_list[i]->_from);
        free(tensor_list[i]);
    }

    free(tensor_list);
}

Tensor** topological_sort(Tensor* head, int* out_count)
{
    Tensor** tensor_list = DALLOC(Tensor, MAX_GRAPH_SIZE);
    int topo_count = 0;

    if (!build_topo(head, tensor_list, &topo_count)) {
        reset_visited(head);
        free(tensor_list);
        *out_count = 0;
        return NULL;
    }

    reset_visited(head);
    *out_count = topo_count; 
    
    return tensor_list;
}

int build_topo(Tensor* t, Tensor** tensor_list, int* topo_count)
{
    if (t == NULL || t->visited == 1)
        return 1;

    if (*topo_count >= MAX_GRAPH_SIZE)
        return 0;

    t->visited = 1;

    if (t->_from != NULL)
    {
        for (int i = 0; i < t->_num_from; i++)
        {
            if (!build_topo(t->_from[i], tensor_list, topo_count))
                return 0;
        }
    }

    if (*topo_count >= MAX_GRAPH_SIZE)
        return 0;

    tensor_list[*topo_count] = t;
    (*topo_count)++;

    return 1;
}

void backward(Tensor* head)
{
    int tensorCount = 0;
    Tensor** tensor_list = topological_sort(head, &tensorCount);
    if (tensor_list == NULL)
        return;

    for (int i = 0; i < tensorCount; i++)
    {
        tensor_list[i]->grad = 0.0f;
    }

    head->grad = 1.0f;

    for (int i = tensorCount - 1; i >= 0; i--)
    {
        tensor_backward(tensor_list[i]);
    }

    free(tensor_list);
}

void tensor_backward(Tensor* t)
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

void reset_visited(Tensor* t)
{
    if (t == NULL || t->visited == 0)
    {
        return;
    }

    t->visited = 0;

    if (t->_from != NULL)
    {
        for (int i = 0; i < t->_num_from; i++)
        {
            reset_visited(t->_from[i]);
        }
    }
}
