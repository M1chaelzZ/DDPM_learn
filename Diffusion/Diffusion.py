
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

'''
定义一个函数 extract，用于在指定的时间步提取系数，并将其重塑为适合广播的形状：

v 是一个向量，包含需要提取的系数。
t 是一个时间步的张量，形状为 [batch_size]。
x_shape 是输入张量 x 的形状，用于确定输出的形状。
device 获取 t 所在的设备（CPU 或 GPU）。
torch.gather 从 v 中提取索引为 t 的元素，结果形状为 [batch_size]。
out.view 将结果重塑为 [batch_size, 1, 1, 1, ...]，其中 1 的个数为 len(x_shape) - 1，用于广播。
'''
def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    '''
    定义一个类 GaussianDiffusionTrainer，用于训练高斯扩散模型：
    model 是一个神经网络模型，用于预测噪声。
    beta_1 和 beta_T 是扩散过程的起始和结束噪声系数。
    T 是扩散过程的时间步数。
    '''
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T
        '''
        具体来说，self.register_buffer 方法有两个参数：
        'betas'：这是缓冲区的名称，类型为字符串。
        torch.linspace(beta_1, beta_T, T).double()：这是要注册的缓冲区的值，类型为张量。
        torch.linspace(beta_1, beta_T, T).double() 是一个函数调用，生成一个从 beta_1 到 beta_T 的线性间隔的张量，
        长度为 T，并将其转换为 double 类型。
        '''
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        '''
        alphas 是 1 - betas，形状为 [T]。
        alphas_bar 是 alphas 的累积乘积，形状为 [T]。
        '''
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        '''
        注册缓冲区 sqrt_alphas_bar，包含 alphas_bar 的平方根，形状为 [T]。
        注册缓冲区 sqrt_one_minus_alphas_bar，包含 1 - alphas_bar 的平方根，形状为 [T]。
        '''
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        """
        Algorithm 1.
        """
        # sample a time step
        '''
        x_0： 原始的清晰图像数据，形状为 [batch_size, ...]。
        t： 随机采样的时间步张量，形状为 [batch_size]，取值范围在 [0, T)。
        noise： 与 x_0 形状相同的标准正态分布噪声张量。
        x_t： 在时间步 t 下的带噪声数据，通过将原始图像 x_0 和噪声 noise 按权重线性组合得到
        torch.randint(self.T, ...)：生成在 [0, self.T) 范围内的随机整数，其中 self.T 是上限（不包括在内）。
        size=(x_0.shape[0], )：指定生成的张量的形状。
        这里 size 是一个元组，表示生成的张量是一维的，其长度等于 x_0 的第一个维度（即 batch_size）。
        '''
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        #noise 是与 x_0 形状相同的标准正态分布噪声张量。
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        # 计算损失
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss

'''
重要变量的形状总结
self.betas：[T]，每个时间步的噪声添加速度。

alphas：[T]，每个时间步的 α。

alphas_bar：[T]，累积的 α。

alphas_bar_prev：[T]，前一个时间步的累积 α。

self.coeff1：[T]，采样过程中的系数。

self.coeff2：[T]，采样过程中的系数。

self.posterior_var：[T]，后验方差。

x_t：[batch_size, channels, height, width]，当前时间步的图像数据。

t：[batch_size]，当前的时间步。

mean：[batch_size, channels, height, width]，前一时刻的均值。

var：[batch_size, channels, height, width]，前一时刻的方差。

eps：[batch_size, channels, height, width]，模型预测的噪声。

noise：[batch_size, channels, height, width]，采样的噪声。 
'''

class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())#betas
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
    
    '''
    作用：根据当前时刻的 x_t、时间步 t 和模型预测的噪声 eps，计算前一时刻 x_{t-1} 的均值。
    参数：
        x_t：当前时刻的图像数据，形状为 [batch_size, channels, height, width]。
        t：当前的时间步，形状为 [batch_size]。
        eps：模型预测的噪声，形状与 x_t 相同。
    变量：
        使用 extract 函数从系数中提取对应时间步的值，并调整形状以与 x_t 匹配。
        计算公式对应于： $$ \mu_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}t}} \boldsymbol{\epsilon}\theta(\mathbf{x}_t, t) \right) $$
    '''
    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):#计算均值
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )


    '''
    作用：计算前一时刻的均值和方差。
    参数：
    x_t：当前时刻的图像数据，形状为 [batch_size, channels, height, width]。
    t：当前的时间步，形状为 [batch_size]。
    变量：
    计算方差 var：
    取 self.posterior_var 的第一个元素（对应时间步为 1）和 self.betas（从第 1 个元素开始），拼接成一个新的张量。
    var：形状为 [T]，包含每个时间步对应的方差。
    通过 extract 函数提取对应时间步的方差值，调整形状为 [batch_size, 1, 1, 1]。
    模型预测噪声 eps：
    eps = self.model(x_t, t)：模型预测的噪声，形状为 [batch_size, channels, height, width]。
    计算前一时刻的均值 xt_prev_mean：
    调用前面的函数 predict_xt_prev_mean_from_eps，得到形状为 [batch_size, channels, height, width]。
    '''
    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        eps = self.model(x_t, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        return xt_prev_mean, var



    '''
    作用：实现从纯噪声 x_T 开始，逐步生成数据的过程，对应于论文中的算法2。
    参数：
    x_T：初始噪声图像，形状为 [batch_size, channels, height, width]，元素服从标准正态分布。
    过程解析：
    初始化：
    x_t = x_T：初始化 x_t 为 x_T。
    循环遍历时间步：
    for time_step in reversed(range(self.T))：从 T-1 到 0，逆序遍历时间步。
    打印当前时间步：print(time_step)。
    创建当前时间步的张量 t：
    [t = x_t.new_ones([x_T.shape[0]], dtype=torch.long) * time_step](http://vscodecontentref/50)
    t 的形状为 [batch_size]，每个元素都是当前的时间步。
    计算均值和方差：
    mean, var = self.p_mean_variance(x_t=x_t, t=t)
    mean 和 var 的形状均为 [batch_size, channels, height, width]。
    采样噪声：
    如果 time_step > 0，则 noise = torch.randn_like(x_t)，形状为 [batch_size, channels, height, width]。
    如果 time_step == 0，则 noise = 0。
    更新 x_t：
    x_t = mean + torch.sqrt(var) * noise
    根据条件概率分布采样得到 x_{t-1}。
    检查是否有 NaN：
    assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
    确保没有数值溢出或计算错误。
    最终输出：
    x_0 = x_t，此时 x_t 已经更新到 t=0，即生成的图像。
    return torch.clip(x_0, -1, 1)：将输出裁剪到 [-1, 1] 的范围内。
    '''
    def forward(self, x_T):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)   


 