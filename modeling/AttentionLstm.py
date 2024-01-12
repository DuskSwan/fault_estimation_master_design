import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 定义结合了注意力的LSTM时间序列预测网络
class LSTMAttentionNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, output_len,
                 num_layers=1,):
        super(LSTMAttentionNet, self).__init__()
        self.output_len = output_len
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True)
        # 注意力机制用于计算每个时间步的每个隐藏状态的权重
        self.attention = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size * output_len)

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        lstm_out, _ = self.lstm(x)  # LSTM层输出
        # lstm_out: [batch_size, seq_len, hidden_size]

        # 为每个隐藏状态计算独立的权重
        attention_weights = torch.softmax(self.attention(lstm_out), dim=2)
        # attention_weights: [batch_size, seq_len, hidden_size]

        # 计算上下文向量
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        # context_vector: [batch_size, hidden_size]

        # 全连接层输出预测结果
        out = self.fc(context_vector)
        # out: [batch_size, output_size * output_len]
        return out.view(-1, self.output_len, self.output_size)

# 使用三角函数产生数据进行测试
def generate_trigonometric_series(n=100, noise=0.1):
    np.random.seed(0)
    x = np.linspace(-2*np.pi, 2*np.pi, n)
    y = np.sin(x) + np.random.normal(0, noise, n)
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
    seq_len = x_len + y_len + n_samples

    # 准备数据
    s = generate_trigonometric_series(seq_len,)
    X, Y = make_X_Y_with_series(s, x_len, y_len, n_samples, seed)
    print('X shape:', X.shape)
    print('Y shape:', Y.shape)
    X = torch.tensor(X, dtype=torch.float32).view(-1, x_len, 1)  # (batch, in_len, input_size)
    Y = torch.tensor(Y, dtype=torch.float32).view(-1, y_len, 1)  # (batch, out_len, input_size)

    # 定义模型
    model = LSTMAttentionNet(input_size=1, hidden_size=10, output_size=1, output_len=5)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

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
        predicted = model(X).detach().numpy()

    # 绘制结果
    x = X[0, :, 0].numpy()
    y = Y[0, :, 0].numpy()
    pred_y = predicted[0, :, 0]
    
    true_s = np.concatenate([x, y])
    pred_s = np.concatenate([x, pred_y])

    plt.plot(true_s, label='Actual')
    plt.plot(pred_s, label='Predicted')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test_net()
