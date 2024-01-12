# encoding: utf-8

# from .example_model import ResNet18
from .LSTM import LSTM_net
from .AttentionLstm import LSTMAttentionNet as LAN

def build_model(cfg, kwargs):
    required_params = ['input_dim', 'output_dim', 'input_len', 'output_len']
    for param in required_params:
        assert param in kwargs and kwargs[param] is not None, f"Missing or None value for parameter: {param}"
    
    if cfg.MODEL.NAME == 'LSTM':
        model = LSTM_net(input_len = kwargs['input_len'], 
                         output_len = kwargs['output_len'],
                         input_dim = kwargs['input_dim'],
                         output_dim = kwargs['output_dim'],
                         lstm_hidden_dim = cfg.MODEL.LSTM_HIDDEN, 
                         line_hidden_dim = cfg.MODEL.LINE_HIDDEN, 
                         used_layers = cfg.MODEL.USED_LAYERS)
    elif cfg.MODEL.NAME == 'LAN':
        model = LAN(input_size = kwargs['input_dim'],
                    hidden_size = cfg.MODEL.LSTM_HIDDEN,
                    output_size = kwargs['output_dim'],
                    output_len = kwargs['output_len'],
                    num_layers = cfg.MODEL.USED_LAYERS)

    if cfg.DATA_TYPE == 'float':
        model = model.float()
    elif cfg.DATA_TYPE == 'double':
        model = model.double()
    return model.to(cfg.DEVICE)
