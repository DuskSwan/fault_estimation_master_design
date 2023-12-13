from torch.nn import Module

class LSTM_net(Module): #继承了父类Module，通用模型
    """
        Parameters：
        - input_len: 每个输入样本的序列长度
        - output_len: 每个样本的输出序列长度
        - input_dim: 输入xt的单点通道数
        - hidden_dim: 隐藏层ht的单点通道数
        - output_dim: 每个样本输出的维数
        - num_layers: LSTM的使用次数/层数
    """
    def __init__(self, input_len, output_len=1, 
                 input_dim=1, output_dim=1, 
                 lstm_hidden_dim=10, line_hidden_dim=16, used_layers=1):
        super().__init__() #调用父类的初始化函数
        self.output_len = output_len
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, used_layers,
                            batch_first=True) #声明了第一维是batch_size
        self.f1 = nn.Linear(input_len*lstm_hidden_dim, line_hidden_dim) 
        self.relu = nn.ReLU()
        self.f2 = nn.Linear(line_hidden_dim, output_len*output_dim)
    def forward(self, _x): #定义了传播的过程
        x, _ = self.lstm(_x)  
            # _x是初始输入，规模(batch, seq_len, input_dim)
            # 其中input_dim是输入的通道数，seq_len即输入长度
            # 过程中还有 hn,cn即隐藏状态和单元状态，形状 layer_n * seq_len * h_dim
        b, s, h = x.shape  
            # x规模(batch, seq_len, hidden_dim),h此时是隐藏层输出的通道数
        x = x.reshape(b,-1) # 将每个样本内部的输出全部拉平
        x = self.f1(x) #计算得到每个样本对应的输出，规模batch*output_dim
        x = self.relu(x)
        x = self.f2(x)
        x = x.reshape(b, self.output_len, -1) #规模(batch, ouput_len, output_dim)
        return x