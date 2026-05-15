#include "neuralnetwork.h"
#include <stdio.h>
#include <time.h>

int main() {
    srand(42);
    
    // Create MLP: 3 inputs, layers of 4, 4, and 1 output
    int layers[] = {4, 4, 1};
    MLP* mlp = mlp_create(3, layers, 3);
    
    // Dataset: 4 examples, 3 features each
    float x_train[4][3] = {
        {2.0f, 3.0f, -1.0f},
        {3.0f, -1.0f, 0.5f},
        {0.5f, 1.0f, 1.0f},
        {1.0f, 1.0f, -1.0f}
    };
    float y_train[4] = {1.0f, -1.0f, -1.0f, 1.0f};
    
    int param_count = 0;
    Tensor** params = mlp_params(mlp, &param_count);
    printf("Total parameters: %d\n", param_count);
    
    float learning_rate = 0.01f;
    
    for (int epoch = 0; epoch < 20; epoch++) {
        float total_loss_val = 0.0f;
        
        for (int i = 0; i < 4; i++) {
            // Forward pass
            Tensor* x[3];
            for(int j=0; j<3; j++) x[j] = tensor_create(x_train[i][j]);
            
            int out_count = 0;
            Tensor** outputs = mlp_forward(mlp, x, 3, &out_count);
            
            // Loss: (output - target)^2
            Tensor* target = tensor_create(y_train[i]);
            Tensor* diff = tensor_sub(outputs[0], target);
            Tensor* loss = tensor_mul(diff, diff);
            
            total_loss_val += loss->data;
            
            // Backward pass
            backward(loss);
            
            // Update weights (SGD)
            for (int p = 0; p < param_count; p++) {
                params[p]->data -= learning_rate * params[p]->grad;
            }

            tensor_free_all(loss);
            free(outputs);
        }
        
        printf("Epoch %d, Loss: %f\n", epoch, total_loss_val);
    }
    
    free(params);
    
    return 0;
}
