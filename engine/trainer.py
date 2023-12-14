# encoding: utf-8

from loguru import logger

import torch
from ignite.engine import Events
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import Loss, RunningAverage


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
):
    log_period = cfg.LOG.ITER_INTERVAL
    checkpoint_period = cfg.TRAIN.CHECKPOINT_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS
    eval_metrics = {'RegLoss': Loss(loss_fn)}

    logger.info(f"Start training with parameters:\n"
                f"max epochs:{epochs}\n"
                f"train batchs:{len(train_loader)}\n")


    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(model, metrics=eval_metrics, device=device)
    checkpointer = ModelCheckpoint(output_dir, "checkpoint", n_saved=10, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.ITERATION_STARTED(every=checkpoint_period), checkpointer, {'model': model})
    # trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model.state_dict(),
    #                                                                  'optimizer': optimizer.state_dict()})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    RunningAverage(output_transform = lambda x: x).attach(trainer, 'avg_loss')

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
                        .format(engine.state.epoch, iter, len(train_loader), engine.state.metrics['avg_loss']))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_loss = metrics['RegLoss']
        logger.info("Training Results - Epoch: {} Avg Loss: {:.6f}"
                    .format(engine.state.epoch, avg_loss))

    if val_loader is not None:
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            avg_loss = metrics['RegLoss']
            logger.info("Validation Results - Epoch: {} Avg Loss: {:.6f}"
                        .format(engine.state.epoch, avg_loss)
                        )

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        timer.reset()
    
    # clear cuda cache
    # @trainer.on(Events.EPOCH_COMPLETED)
    # def clear_cuda_cache(engine):
    #     torch.cuda.empty_cache()

    trainer.run(train_loader, max_epochs=epochs)
