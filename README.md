# otograd

C için yazılan ve Andrej Karpathy'nin "micrograd" projesinden ve bu projeyi tekrardan yazdığı "The spelled-out intro to neural networks and backpropagation: building micrograd" isimli videosundan yola çıkılarak, benim tarafımdan hazırlanan küçük bir autograd kütüphanesidir.

## Özellikler

- Skaler `Tensor` oluşturma
- `+`, `*`, `-`, `/` işlemleri için düğüm oluşturma

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

## Notlar

- Kütüphane yalnızca skaler değerler için çalışır.
- Hatalar bulunabilir.
