# encoding: utf-8

from loguru import logger

from ignite.engine import Events
from ignite.engine import create_supervised_evaluator
from ignite.metrics import MeanAbsoluteError

from torch import abs as tensor_abs
import torch

def inference(cfg, model, val_loader):
    device = cfg.DEVICE
    model.eval()
    logger.info("Start inferencing")

    # Check input dimensions
    # input_dim = val_loader.dataset[0][0].size()
    # model_input_dim = model.input_size
    # assert input_dim == model_input_dim, f"Input dimensions {input_dim} do not match model's input dimensions {model_input_dim}"

    evaluator = create_supervised_evaluator(model, metrics={'MAE': MeanAbsoluteError()}, device=device)

    # record per batch MAE
    batch_scores = [] # 其中每个元素会是一个tensor数组
    @evaluator.on(Events.ITERATION_COMPLETED)
    def save_batch_score(engine):
        y_pred, y = engine.state.output
        batch_score = tensor_abs(y_pred - y).mean()
        batch_scores.append(batch_score)

    evaluator.run(val_loader)
    return batch_scores
