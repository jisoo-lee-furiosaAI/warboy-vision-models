#include <torch/extension.h>
#include <vector>

__global__ void draw_boxes_kernel(
    uint8_t* image,        // (3,H,W)
    const float* boxes,    // (N,4)
    int num_boxes,
    int H,
    int W,
    uint8_t r, uint8_t g, uint8_t b
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boxes) return;

    const float* box = boxes + idx * 4;
    int x1 = max(0, min((int)box[0], W - 1));
    int y1 = max(0, min((int)box[1], H - 1));
    int x2 = max(0, min((int)box[2], W - 1));
    int y2 = max(0, min((int)box[3], H - 1));

    // Draw rectangle
    for (int x = x1; x <= x2; ++x) {
        int top_idx = (y1 * W + x);
        int bottom_idx = (y2 * W + x);
        image[top_idx] = r;
        image[top_idx + H * W] = g;
        image[top_idx + 2 * H * W] = b;

        image[bottom_idx] = r;
        image[bottom_idx + H * W] = g;
        image[bottom_idx + 2 * H * W] = b;
    }
    for (int y = y1; y <= y2; ++y) {
        int left_idx = (y * W + x1);
        int right_idx = (y * W + x2);
        image[left_idx] = r;
        image[left_idx + H * W] = g;
        image[left_idx + 2 * H * W] = b;

        image[right_idx] = r;
        image[right_idx + H * W] = g;
        image[right_idx + 2 * H * W] = b;
    }
}

void draw_boxes_cuda(torch::Tensor image, torch::Tensor boxes,
                     uint8_t r, uint8_t g, uint8_t b) {
    const int num_boxes = boxes.size(0);
    const int H = image.size(1);
    const int W = image.size(2);

    const int threads = 256;
    const int blocks = (num_boxes + threads - 1) / threads;

    draw_boxes_kernel<<<blocks, threads>>>(
        image.data_ptr<uint8_t>(),
        boxes.data_ptr<float>(),
        num_boxes,
        H, W, r, g, b
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("draw_boxes_cuda", &draw_boxes_cuda, "Draw boxes (CUDA)");
}

