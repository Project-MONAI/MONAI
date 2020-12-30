
#include <torch/extension.h>

// Struct to easily cache descriptive information about a tensor.
// This is helpful as regular calls to the size and stride member
// functions of tensors appear to cause memory issues.
struct TensorDescription {
 public:
  TensorDescription(torch::Tensor tensor) {
    batchCount = tensor.size(0);
    batchStride = tensor.stride(0);

    channelCount = tensor.size(1);
    channelStride = tensor.stride(1);

    dimensions = tensor.dim() - 2;
    sizes = new int[dimensions];
    strides = new int[dimensions];

    for (int i = 0; i < dimensions; i++) {
      sizes[i] = tensor.size(i + 2);
      strides[i] = tensor.stride(i + 2);
    }
  }

  ~TensorDescription() {
    delete[] sizes;
    delete[] strides;
  }

  int batchCount;
  int batchStride;

  int channelCount;
  int channelStride;

  int dimensions;
  int* sizes;
  int* strides;
};
