import torch

x = torch.ones(2, 2, requires_grad=True)
y1 = x * 2
y2 = y1 * y1
y3 = y2 * y2 * y2
print(x.requires_grad, y1.requires_grad, y2.requires_grad, y3.requires_grad)
x.requires_grad = False
print(x.requires_grad, y1.requires_grad, y2.requires_grad, y3.requires_grad)
