import torch
import torch.nn as nn#nn：neutral network
import torch.nn.functional
class Flow_Loss(nn.Module):#计算光流损失
    def __init__(self):#__init__函数的参数列表会在开头多出一项，它永远指代新建的那个实例对象，Python语法要求这个参数必须要有，而名称随意，习惯上就命为self。
        super().__init__()#调用构造函数

    def forward(self, gen_flows, gt_flows):#前向传播函数forward方法 后面是自己定义的属性
        return torch.mean(torch.abs(gen_flows - gt_flows))#输出input 各个元素的的均值


class Adversarial_Loss(nn.Module):#生成对抗损失
    def __init__(self):
        super().__init__()

    def forward(self, fake_outputs):
        # TODO: compare with torch.nn.MSELoss ?
        return torch.mean((fake_outputs - 1) ** 2 / 2)


class Discriminate_Loss(nn.Module):#鉴别器损失
    def __init__(self):
        super().__init__()

    def forward(self, real_outputs, fake_outputs):
        return torch.mean((real_outputs - 1) ** 2 / 2) + torch.mean(fake_outputs ** 2 / 2)