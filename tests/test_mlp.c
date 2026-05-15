#include "../neuralnetwork/neuralnetwork.h"
#include <stdio.h>
#include <time.h>

int main() {
    srand(42);
    int layers[] = {4, 4, 1};
    MLP* mlp = mlp_create(3, layers, 3);
    
    Tensor* x[3];
    float x_data[] = {2.0f, 3.0f, -1.0f};
    for(int i=0; i<3; i++) x[i] = tensor_create(x_data[i]);

    int out_count = 0;
    Tensor** outputs = mlp_forward(mlp, x, 3, &out_count);
    
    printf("MLP output: %f\n", outputs[0]->data);
    
    int params = mlp_param_count(mlp);
    printf("MLP parameter count: %d\n", params);
    
    // Cleanup
    tensor_free_all(outputs[0]);
    free(outputs);
    
    return 0;
}
