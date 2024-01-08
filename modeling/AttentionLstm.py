import torch
import torch.nn as nn
import torch.optim as optim

# 定义LSTM编码器
class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(EncoderLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell

# 定义注意力机制
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, encoder_outputs, decoder_hidden):
        seq_len = encoder_outputs.size(1)
        decoder_hidden = decoder_hidden.repeat(seq_len, 1, 1).transpose(0, 1)
        cat_hidden = torch.cat((decoder_hidden, encoder_outputs), dim=2)
        energy = torch.tanh(self.attention(cat_hidden))
        attention_weights = torch.softmax(energy.squeeze(2), dim=1)
        return attention_weights

# 定义LSTM解码器
class DecoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(DecoderLSTM, self).__init__()
        self.lstm = nn.LSTM(hidden_size + input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell

# 定义LSTM-Attention-LSTM模型
class LSTMAttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMAttentionLSTM, self).__init__()
        self.encoder = EncoderLSTM(input_size, hidden_size, num_layers)
        self.decoder = DecoderLSTM(input_size + hidden_size, hidden_size, output_size, num_layers)
        self.attention = Attention(hidden_size)
        self.input_size = input_size

    def forward(self, source, target_len):
        encoder_outputs, hidden, cell = self.encoder(source)
        batch_size = source.size(0)
        decoder_input = torch.zeros(batch_size, 1, self.input_size).to(source.device)

        outputs = torch.zeros(batch_size, target_len, self.decoder.fc.out_features).to(source.device)

        for t in range(target_len):
            attention_weights = self.attention(encoder_outputs, hidden[-1])
            context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
            decoder_input = torch.cat((context.unsqueeze(1), decoder_input), dim=2)
            out, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t] = out.squeeze(1)

        return outputs

# 生成模拟数据
def generate_random_data(batch_size, seq_len, input_size):
    return torch.rand(batch_size, seq_len, input_size)

def generate_log_data(batch_size, seq_len, input_size):
    x = torch.linspace(1, 10, steps=seq_len)
    y = torch.log(x)
    return y.repeat(batch_size, 1, input_size)

# 测试模型
def test():
    seq_len = 10
    input_size = 1
    batch_size = 32

    data = generate_log_data(batch_size, seq_len, input_size)
    input_seq = data[:, :seq_len // 2, :]
    target_seq = data[:, seq_len // 2:, :]

    train_size = int(batch_size * 0.8)
    train_input, test_input = input_seq[:train_size], input_seq[train_size:]
    train_target, test_target = target_seq[:train_size], target_seq[train_size:]

    hidden_size = 128
    output_size = 1
    num_layers = 1

    model = LSTMAttentionLSTM(input_size, hidden_size, output_size, num_layers)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_input, train_target.shape[1])
        loss = criterion(output, train_target)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item()}')

    model.eval()
    with torch.no_grad():
        predicted = model(test_input, test_target.shape[1])
        test_mae = criterion(predicted, test_target)
        print(f'Test MAE: {test_mae.item()}')

if __name__ == '__main__':
    test()
