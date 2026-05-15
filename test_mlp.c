#include "neuralnetwork.h"
#include <stdio.h>
#include <time.h>

int main() {
    srand(42); // Fixed seed for reproducibility
    int layers[] = {4, 4, 1};
    MLP* mlp = mlp_create(3, layers, 3);
    
    float x[] = {2.0f, 3.0f, -1.0f};
    float out = mlp_forward(mlp, x, 3);
    
    printf("MLP output: %f\n", out);
    
    int params = mlp_param_count(mlp);
    printf("MLP parameter count: %d\n", params);
    
    return 0;
}
