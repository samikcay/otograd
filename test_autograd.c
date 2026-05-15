#include "otograd.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

int main() {
    Tensor* a = tensor_create(2.0f);
    Tensor* b = tensor_create(-3.0f);
    Tensor* c = tensor_create(10.0f);
    
    // e = a * b + c
    Tensor* d = tensor_mul(a, b);
    Tensor* e = tensor_add(d, c);
    
    printf("e data: %f (expected 4.0)\n", e->data);
    
    backward(e);
    
    printf("a grad: %f (expected -3.0)\n", a->grad);
    printf("b grad: %f (expected 2.0)\n", b->grad);
    printf("c grad: %f (expected 1.0)\n", c->grad);

    tensor_free_all(e);
    
    return 0;
}
