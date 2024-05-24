# encoding: utf-8

from loguru import logger

# import torch
from ignite.engine import Events
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import Loss, RunningAverage

import matplotlib.pyplot as plt

def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
):
    device = cfg.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS
    eval_metrics = {'RegLoss': Loss(loss_fn)}

    logger.info(f"Start training with parameters: max epochs:{epochs} train batches:{len(train_loader)}")

    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(model, metrics=eval_metrics, device=device)

    if(cfg.TRAIN.NEED_CHECKPOINT):
        output_dir = cfg.OUTPUT.MODEL_DIR
        checkpoint_period = cfg.TRAIN.CHECKPOINT_PERIOD
        checkpointer = ModelCheckpoint(output_dir, "checkpoint", n_saved=5, require_empty=False)
        trainer.add_event_handler(Events.ITERATION_STARTED(every=checkpoint_period), checkpointer, {'model': model})
    
    timer = Timer(average=True)
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    RunningAverage(output_transform = lambda x: x).attach(trainer, 'avg_loss')

    train_losses = []
    val_losses = []

    @trainer.on(Events.ITERATION_COMPLETED(every=cfg.LOG.ITER_INTERVAL))
    def log_training_loss(engine):
        iter_n = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter_n % cfg.LOG.ITER_INTERVAL == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
                        .format(engine.state.epoch, iter_n, len(train_loader), engine.state.metrics['avg_loss']))

    @trainer.on(Events.EPOCH_COMPLETED(every=cfg.LOG.EPOCH_INTERVAL))
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_loss = metrics['RegLoss']
        train_losses.append(avg_loss)
        logger.info("Training Results - Epoch: {} Avg Loss: {:.6f}"
                    .format(engine.state.epoch, avg_loss))

    if val_loader is not None:
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            avg_loss = metrics['RegLoss']
            val_losses.append(avg_loss)
            logger.info("Validation Results - Epoch: {} Avg Loss: {:.6f}"
                        .format(engine.state.epoch, avg_loss))

    # @trainer.on(Events.EPOCH_COMPLETED)
    # def print_times(engine):
    #     logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
    #                 .format(engine.state.epoch, timer.value() * timer.step_count,
    #                         train_loader.batch_size / timer.value()))
    #     timer.reset()
    
    # clear cuda cache
    # @trainer.on(Events.EPOCH_COMPLETED)
    # def clear_cuda_cache(engine):
    #     torch.cuda.empty_cache()

    trainer.run(train_loader, max_epochs=epochs)

    # Plotting training and validation loss
    if cfg.TRAIN.NEED_PLOT_LOSS:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
        if val_loader is not None:
            plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()
