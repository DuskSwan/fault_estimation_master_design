import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
# from einops import rearrange

class Attention(nn.Module):
    """
    Attention module that performs multi-head self-attention.

    Args:
        dim (int): The input dimension.
        heads (int, optional): The number of attention heads. Defaults to 8.
        dim_head (int, optional): The dimension of each attention head. Defaults to 64.
        dropout (float, optional): The dropout rate. Defaults to 0.0.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        '''
        dim: 输入维度
        heads: 多头注意力的头数
        dim_head: 每个头的维度
        dropout: dropout概率
        '''
        super().__init__()
        inner_dim = dim_head * heads # 内部维度，等于头数乘以头维度。
        project_out = not (heads == 1 and dim_head == dim)
            # 这个变量决定了是否需要对输出进行线性变换。当头数为1且头维度等于输入维度时，不需要进行线性变换。

        self.heads = heads
        self.scale = dim_head**-0.5
            # 缩放因子，用于缩放点积的结果，防止梯度消失或爆炸。

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
                # 如果不需要进行线性变换，则直接返回一个恒等映射。
        )

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying multi-head self-attention.
        """
        x = self.norm(x) # (n, l, d) -> (n, l, d)

        qkv = self.to_qkv(x) # (n, l, d) -> (n, l, 3*inner_dim)
        qkv = qkv.chunk(3, dim=-1) # (n, l, 3*inner_dim) -> 3 * (n, l, inner_dim)
        # q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        q, k, v = qkv[0], qkv[1], qkv[2] # (n, l, inner_dim)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            # 计算点积注意力得分。
            # q: (n, l, inner_dim)
            # k: (n, l, inner_dim)
            # q.k -> (n, l, l)
            # dots = q.k / sqrt(d) -> (n, l, l)


        attn = self.attend(dots) # (n, l, l) -> (n, l, l)
        attn = self.dropout(attn) # (n, l, l) -> (n, l, l)

        out = torch.matmul(attn, v) # (n, l, l) * (n, l, inner_dim) -> (n, l, inner_dim)
        # out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out) # (n, l, inner_dim) -> (n, l, d)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        '''
        dim: 输入维度
        depth: 单个Transformer块的层数
        heads: 多头注意力的头数
        dim_head: 每个头的维度
        mlp_dim: MLP的隐藏层维度
        dropout: dropout概率
        '''
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x # (n, l, d) -> (n, l, d)
            x = ff(x) + x # (n, l, d) -> (n, l, d)
        return self.norm(x)


# 定义结合了注意力的LSTM时间序列预测网络
class LSTMAttentionNet(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size, output_size, output_len,
                 num_layers=1,):
        super(LSTMAttentionNet, self).__init__()
        self.output_len = output_len
        self.output_size = output_size
        self.transformer = Transformer(input_size, 
                                       depth=1, heads=8, 
                                       dim_head=64, 
                                       mlp_dim=128, 
                                       dropout=0.0)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True)
        self.fc = nn.Linear(seq_len * hidden_size, output_size * output_len)
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_size] 

        x = self.transformer(x) 
            # (batch_size, seq_len, input_size) -> (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)  # LSTM层输出
            # (batch_size, seq_len, input_size) -> (batch_size, seq_len, hidden_size)
        batch_size = lstm_out.size(0)
        c = lstm_out.reshape(batch_size, -1) # 将每个样本内部的输出全部拉平
            # (batch_size, seq_len, hidden_size) -> (batch_size, seq_len * hidden_size)
        out = self.fc(c)
            # (batch_size, seq_len * hidden_size) -> (batch_size, output_len*output_size)
        out = out.reshape(-1, self.output_len, self.output_size)
            # (batch_size, output_len*output_size) -> (batch_size, output_len, output_size)
        return out

# 使用三角函数产生数据进行测试
def generate_trigonometric_series(n=100, noise=0.1):
    np.random.seed(0)
    x = np.linspace(-2*np.pi, 2*np.pi, n)
    y = 3*np.sin(x) + np.random.normal(0, noise, n)
    return y

def make_X_Y_with_series(s, x_len, y_len, n_samples=100, seed=0):
    np.random.seed(seed)
    X, Y = [], []
    for _ in range(n_samples):
        start = np.random.randint(0, len(s) - x_len - y_len)
        x = s[start:start+x_len]
        y = s[start+x_len:start+x_len+y_len]
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)

def test_net():
    # 设置随机种子
    seed = 0
    torch.manual_seed(seed)

    # 设置参数
    x_len = 200
    y_len = 5
    n_samples = 2000
    max_len = x_len + y_len + n_samples

    # 准备数据
    s = generate_trigonometric_series(max_len)
    X, Y = make_X_Y_with_series(s, x_len, y_len, n_samples, seed)
    print('X shape:', X.shape)
    print('Y shape:', Y.shape)
    X = torch.tensor(X, dtype=torch.float32).view(-1, x_len, 1)  # (batch, in_len, input_size)
    Y = torch.tensor(Y, dtype=torch.float32).view(-1, y_len, 1)  # (batch, out_len, input_size)

    # 定义模型
    model = LSTMAttentionNet(input_size=1, seq_len=x_len, hidden_size=10, output_size=1, output_len=y_len)
    
    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device',device)
    model.to(device)
    X, Y = X.to(device), Y.to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练
    for epoch in range(100):
        output = model(X)
        loss = criterion(output, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print('Epoch[{}/{}], loss: {:.6f}'.format(epoch+1, 100, loss.item()))

    # 预测
    with torch.no_grad():
        model.eval()
        predicted = model(X).cpu().detach().numpy()

    # 绘制结果
    x = X[0, :, 0].cpu().numpy()
    y = Y[0, :, 0].cpu().numpy()
    pred_y = predicted[0, :, 0]
    
    true_s = np.concatenate([x, y])
    pred_s = np.concatenate([x, pred_y])

    plt.plot(true_s, label='Actual')
    plt.plot(pred_s, label='Predicted')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test_net()
