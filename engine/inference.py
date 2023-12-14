# encoding: utf-8

from loguru import logger

from ignite.engine import Events
from ignite.engine import create_supervised_evaluator
from ignite.metrics import MeanAbsoluteError


def inference(cfg,model,val_loader):
    device = cfg.DEVICE
    logger.info("Start inferencing")
    evaluator = create_supervised_evaluator(model, metrics={'MAE': MeanAbsoluteError()},
                                            device=device)

    # view the validation score
    @evaluator.on(Events.EPOCH_COMPLETED)
    def print_validation_results(engine):
        metrics = evaluator.state.metrics
        avg_score = metrics['MAE']
        logger.info("Validation Results - MAE: {:.3f}".format(avg_score))

    # record per batch MAE
    batch_scores = [] # 其中每个元素会是一个tensor数组
    @evaluator.on(Events.ITERATION_COMPLETED)
    def save_batch_score(engine):
        batch_score = engine.state.output
        batch_scores.append(batch_score)

    evaluator.run(val_loader)
    return batch_scores
