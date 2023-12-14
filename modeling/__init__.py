# encoding: utf-8

# from .example_model import ResNet18
from .LSTM import LSTM_net

def build_model(cfg, kwargs):
    required_params = ['input_dim', 'output_dim', 'input_len', 'output_len']
    for param in required_params:
        assert param in kwargs and kwargs[param] is not None, f"Missing or None value for parameter: {param}"
    
    model = LSTM_net(input_len = kwargs['input_len'], 
                     output_len = kwargs['output_len'],
                     input_dim = kwargs['input_dim'],
                     output_dim = kwargs['output_dim'],
                     lstm_hidden_dim = cfg.MODEL.LSTM_HIDDEN, 
                     line_hidden_dim = cfg.MODEL.LINE_HIDDEN, 
                     used_layers = cfg.MODEL.USED_LAYERS) 
    return model
