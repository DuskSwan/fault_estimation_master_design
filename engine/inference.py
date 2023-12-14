# encoding: utf-8

from loguru import logger

from ignite.engine import Events
from ignite.engine import create_supervised_evaluator
from ignite.metrics import MeanSquaredError


def inference(cfg,model,val_loader):
    device = cfg.MODEL.DEVICE

    logger.info("Start inferencing")
    evaluator = create_supervised_evaluator(model, metrics={'MSE': MeanSquaredError()},
                                            device=device)

    # adding handlers using `evaluator.on` decorator API
    @evaluator.on(Events.EPOCH_COMPLETED)
    def print_validation_results(engine):
        metrics = evaluator.state.metrics
        avg_score = metrics['MSE']
        logger.info("Validation Results - MSE: {:.3f}".format(avg_score))

    evaluator.run(val_loader)
