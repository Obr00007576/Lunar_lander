import torch
import math

eps=1e-7

w12 = torch.randn(9, 40)*math.sqrt(1/9)
w23 = torch.randn(40, 50)*math.sqrt(1/40)
w34 = torch.randn(50, 1)*math.sqrt(1/50)

b12 = torch.randn(1, 40)*math.sqrt(1/40)
b23 = torch.randn(1, 50)*math.sqrt(1/50)

w12.requires_grad=True
w23.requires_grad=True
w34.requires_grad=True

b12.requires_grad=True
b23.requires_grad=True

def get_output(l1):
    global w12, w23, w34, b12, b23
    l2 = torch.tanh((torch.matmul(l1, w12))+b12)
    l3 = torch.sigmoid(torch.matmul(l2, w23)+b23)
    l4 = torch.matmul(l3, w34)
    return l4

def update(res, target, alpha=0.01):
    global w12, w23, w34, b12, b23
    #loss = -torch.sum(torch.mul(torch.log(res+eps), target))
    loss = (target-res)*(target-res)
    loss.backward(retain_graph=True)

    w12.data-=w12.grad.data*alpha
    w23.data-=w23.grad.data*alpha
    w34.data-=w34.grad.data*alpha

    b12.data-=b12.grad.data*alpha
    b23.data-=b23.grad.data*alpha

    w12.grad.data.zero_()
    w23.grad.data.zero_()
    w34.grad.data.zero_()

    b12.grad.data.zero_()
    b23.grad.data.zero_()

# aaa=0.001
# for i in range(70):
#     update(w12, w23, w34, b12, b23, get_output(torch.FloatTensor([1,1,1,1,1,1,1,2,1]), w12, w23, w34, b12, b23), torch.FloatTensor([aaa]), 0.01)
# print(get_output(torch.FloatTensor([1,1,1,1,1,1,1,2,1]), w12, w23, w34, b12, b23))

# for i in range(20):
#     res = get_output(data, w12, w23, w34, b12, b23)
#     loss = -torch.sum(torch.mul(torch.log(res+eps), target))
#     #loss = torch.sum(torch.abs(target-res))
#     loss.backward()

#     w12.data-=w12.grad.data
#     w23.data-=w23.grad.data
#     w34.data-=w34.grad.data

#     b12.data-=b12.grad.data
#     b23.data-=b23.grad.data

#     w12.grad.data.zero_()
#     w23.grad.data.zero_()
#     w34.grad.data.zero_()

#     b12.grad.data.zero_()
#     b23.grad.data.zero_()

# print(get_output(data, w12, w23, w34, b12, b23))
