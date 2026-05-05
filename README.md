# otograd

C için yazılan ve Andrej Karpathy'nin "micrograd" projesinden ve bu projeyi tekrardan yazdığı "The spelled-out intro to neural networks and backpropagation: building micrograd" isimli videosundan yola çıkılarak, benim tarafımdan hazırlanan küçük bir autograd kütüphanesidir.

## Özellikler

- Skaler `Tensor` oluşturma
- `+`, `*`, `-`, `/` işlemleri için düğüm oluşturma
- Hesap grafiği üzerinde topolojik sıralama
- `backward` ile otomatik gradyan hesabı
- Paylaşılan düğüm içeren grafikleri `tensor_free_all` ile güvenli serbest bırakma
- Basit `Neuron`, `Layer` ve `MLP` yardımcı yapıları

## Örnek Kullanım

```c
#include "otograd.h"

int main(void)
{
    Tensor* a = tensor_create(2.0f);
    Tensor* b = tensor_create(3.0f);
    Tensor* c = tensor_add(a, b);
    Tensor* d = tensor_mul(c, a);
    backward(d);

    // d.data = 10, d.grad = 1
    // c.data = 5, c.grad = 2
    // b.data = 3, b.grad = 2
    // a.data = 2, a.grad = 7

    tensor_free_all(d);
    return 0;
}
```

## Neural Network Örneği

```c
#include "neuralnetwork.h"

int main(void)
{
    int layers[] = {3, 1};
    float xs[] = {0.5f, -0.25f};

    MLP* mlp = mlp_create(2, layers, 2);
    float y = mlp_forward(mlp, xs, 2);

    int param_count = 0;
    Tensor** params = mlp_params(mlp, &param_count);

    // param_count = 13
    // params[0]  = layer 0, neuron 0, w[0]
    // params[1]  = layer 0, neuron 0, w[1]
    // params[2]  = layer 0, neuron 0, b
    // params[3]  = layer 0, neuron 1, w[0]
    // params[4]  = layer 0, neuron 1, w[1]
    // params[5]  = layer 0, neuron 1, b
    // params[6]  = layer 0, neuron 2, w[0]
    // params[7]  = layer 0, neuron 2, w[1]
    // params[8]  = layer 0, neuron 2, b
    // params[9]  = layer 1, neuron 0, w[0]
    // params[10] = layer 1, neuron 0, w[1]
    // params[11] = layer 1, neuron 0, w[2]
    // params[12] = layer 1, neuron 0, b

    free(params);
    return 0;
}
```

## Notlar

- Kütüphane yalnızca skaler değerler için çalışır.
- `backward(head)` çağrısı, grafikteki mevcut gradyanları sıfırlar ve gradyanları yeniden hesaplar.
- `tensor_free_all(head)`, `head` üzerinden erişilebilen hesap grafiğindeki her `Tensor` nesnesini bir kez serbest bırakır.
- Topolojik sıralama için varsayılan sınır `MAX_GRAPH_SIZE` değeridir. Bu sınır aşılırsa `topological_sort` `NULL` döndürür.
- `MLP` tarafında `mlp_forward`, her katmanın çıktısını sonraki katmana aktarır ve son katman çıktılarının toplamını `float` olarak döndürür.
- Hatalar bulunabilir.
