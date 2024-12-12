
   
import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

'''
整体功能说明
这段代码实现了一个用于 扩散模型（Diffusion Model） 的 UNet 网络结构，主要用于图像生成和去噪任务。
模型的结构包括时间步编码（Time Embedding）、残差块（ResBlock）、注意力机制（AttnBlock）、下采样（DownSample）和上采样（UpSample）等模块。

该模型的主要功能是：

接受输入：带有噪声的图像 x 和对应的时间步 t。
时间步编码：将时间步 t 转换为时间嵌入 temb，用于在网络中传递时间信息。
编码过程：通过多层残差块和下采样，对输入图像进行编码，提取多尺度的特征。
瓶颈层：在最低分辨率下进行进一步的特征提取，通常包含注意力机制。
解码过程：通过多层残差块和上采样，逐步恢复图像的空间分辨率，同时融合编码过程中的特征（跳跃连接）。
输出：生成与输入形状相同的图像 y，用于预测去噪后的图像或噪声。
'''

#Sigmoid 激活函数
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    
    
    #T：时间步的总数，即时间序列的长度。
    # d_model：时间嵌入的维度，通常为偶数。
    # dim：线性变换后的维度，用于调整嵌入向量的大小。
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        #生成从 0 到 d_model（不含），步长为 2 的整数序列，除以 d_model 并乘以 math.log(10000)，得到频率序列。
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        #对上述计算结果取负数，然后计算指数函数 exp，得到衰减的频率序列，形状为 [d_model // 2]。
        emb = torch.exp(-emb)
        #生成时间步索引序列，从 0 到 T-1，形状为 [T]，表示每个时间步的位置
        pos = torch.arange(T).float()
        # pos[:, None]：将 pos 调整形状为 [T, 1]，以便进行广播运算。
        # emb[None, :]：将 emb 调整形状为 [1, d_model // 2]。
        # 两者相乘，利用广播机制，得到形状为 [T, d_model // 2] 的矩阵，每个元素表示位置 pos 与频率 emb 的乘积
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        #计算 emb 的正弦和余弦值，分别得到两个形状为 [T, d_model // 2] 的矩阵。
        #使用 torch.stack 将这两个矩阵在最后一个维度上堆叠，得到形状为 [T, d_model // 2, 2] 的张量。
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        #将 emb 重新调整形状为 [T, d_model]，将最后两个维度合并，从而每个时间步对应一个长度为 d_model 的嵌入向量
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            #使用预训练的嵌入矩阵 emb 创建嵌入层，将时间步索引映射到嵌入向量。输入时间步索引 t，输出形状为 [batch_size, d_model]
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()


    # 定义初始化函数 initialize。
    # 遍历模型的所有子模块：
    # 如果模块是线性层 nn.Linear，则对其权重和偏置进行初始化：
    # init.xavier_uniform_(module.weight)：使用 Xavier 均匀分布初始化权重，保持前向传播时方差一致。
    # init.zeros_(module.bias)：将偏置项初始化为零。
    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        # x：输入张量，形状为 [batch_size, in_ch, H, W]。
        # temb：时间嵌入张量（在此模块中未使用，但为了与其他模块的接口保持一致，保留该参数）。
        _, _, H, W = x.shape
        # 对输入张量 x 进行上采样操作：
        # scale_factor=2：将特征图的高度和宽度各扩大 2 倍。
        # mode='nearest'：使用最近邻插值方法，上采样后的像素值取最近的邻近值。
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        # 定义一个 GroupNorm 层，对输入特征进行归一化。
        # num_groups=32：将通道分成 32 组。
        # num_channels=in_ch：输入特征的通道数。
        self.group_norm = nn.GroupNorm(32, in_ch)
        # 定义四个卷积层，用于计算查询、键、值和投影（输出的线性变换）。
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        #对 self.proj 的权重进行特殊初始化，设置增益 gain=1e-5，使权重值更小，有助于稳定训练。
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        #分别通过三个卷积层生成查询（q）、键（k）、值（v）矩阵，形状均为 [B, C, H, W]。
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)
        # 调整 q 的维度顺序和形状：
        # permute(0, 2, 3, 1)：将通道维移动到最后，形状变为 [B, H, W, C]。
        # view(B, H * W, C)：将空间维度展平成一维，形状变为 [B, H*W, C]。
        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        #将 k 展平成二维矩阵，形状为 [B, C, H*W]。
        k = k.view(B, C, H * W)
        # 计算注意力权重矩阵 w：
        # torch.bmm(q, k)：执行矩阵乘法，计算查询和键的点积，形状为 [B, H*W, H*W]。
        # (int(C) ** (-0.5))：将 C 转换为整数，然后取倒数平方根，用于缩放点积。
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        #在最后一个维度对 w 进行 softmax，从而得到注意力权重。
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    # in_ch：输入通道数。
    # out_ch：输出通道数。
    # tdim：时间嵌入的维度。
    # dropout：Dropout 的概率。
    # attn：是否使用注意力机制，默认为 False。
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class UNet(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)

        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):
        # Timestep embedding
        temb = self.time_embedding(t)
        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)

        assert len(hs) == 0
        return h


if __name__ == '__main__':
    batch_size = 8
    model = UNet(
        T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size, ))
    y = model(x, t)
    print(y.shape)

