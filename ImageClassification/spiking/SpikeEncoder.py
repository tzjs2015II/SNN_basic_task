import torch

# 使用 torch.stack 函数将 img_stack 中的所有展平后的图像向量堆叠成一个张量。
def SimpleEncoder(img_batch):
    img_stack = []
    for img in img_batch:
        img_stack.append(img.view(-1))
    return torch.stack(img_stack)
