import torch
from ignite.handlers import Checkpoint, DiskSaver
from .train_handler import TrainHandler


class CheckpointHandler(TrainHandler):
    """Checkpoint handler includes all checkpoint related logics during traing.
    It can restore checkpoint and session to train, and also save checkpoint and sessions.
    supports to save according to validation, epoch number, step number and train_end or exception.

    Args:
        network(torch.nn.Module): the model to save or restore.
        optimizer(torch.nn.optim): the optimizer to save or restore.
        lr_policy(torch.nn.optim): the lr_policy to save or restore.
        restore_ckpt(Bool): whether to restore model, optimizer, lr_policy.
        restore_session(Bool): whether to restore previous epoch, step, etc.
        restore_path(String): the checkpoint file path to restore training.
        save_ckpt(Bool): whether to save model, optimizer, lr_policy.
        save_session(Bool): whether to save current epoch, current step, etc.
        save_path(String): the dir to save models.
        save_val_model(Bool): whether to save model during validation.
        val_n_saved(Int): save top N models sorted by validation metric.
        key_metric_name(String): use which metric to select model.
        epoch_save_interval(Int): save model every N epochs.
        epoch_n_saved(Int): how many models to save, 'None' is to save all models.
        step_save_interval(Int): save model every N intervals.
        step_n_saved(Int): how many models to save, 'None' is to save all models.

    Notes:
        example of saved models:
            checkpoint_step=400.pth
            checkpoint_step=800.pth
            checkpoint_epoch=1.pth
            checkpoint_final_step=1000.pth
            checkpoint_key_metric=0.9387.pth
    """

    def __init__(self,
                 network=None,
                 optimizer=None,
                 lr_policy=None,
                 restore_ckpt=True,
                 restore_session=False,
                 restore_path=None,
                 save_ckpt=True,
                 save_session=False,
                 save_path=None,
                 save_val_model=True,
                 val_n_saved=1,
                 key_metric_name=None,
                 epoch_save_interval=0,
                 epoch_n_saved=None,
                 step_save_interval=0,
                 step_n_saved=None):
        TrainHandler.__init__(self)
        self.network = network
        self.optimizer = optimizer
        self.lr_policy = lr_policy
        if restore_ckpt is True or restore_session is True:
            assert restore_path is not None, 'must set path to restore checkpoint.'
        self.restore_ckpt = restore_ckpt
        self.restore_session = restore_session
        self.restore_path = restore_path
        if save_ckpt is True or save_session is True:
            assert save_path is not None, 'must set path to save checkpoint.'
        self.save_ckpt = save_ckpt
        self.save_session = save_session
        self.save_path = save_path
        if save_val_model is True:
            assert key_metric_name is not None and type(key_metric_name) == str, 'must set key_metric_name.'
        self.save_val_model = save_val_model
        self.val_n_saved = val_n_saved
        self.key_metric_name = key_metric_name
        self.epoch_save_interval = epoch_save_interval
        self.epoch_n_saved = epoch_n_saved
        self.step_save_interval = step_save_interval
        self.step_n_saved = step_n_saved

    def start(self, engine):
        super().start(engine)
        to_restore_dict = dict()
        if self.restore_ckpt is True:
            if self.network is not None:
                to_restore_dict.update({"model": self.network})
            if self.optimizer is not None:
                to_restore_dict.update({"optimizer": self.optimizer})
            if self.lr_policy is not None:
                to_restore_dict.update({"lr_scheduler": self.lr_policy})
        if self.restore_session is True:
            to_restore_dict.update({'engine': engine})
        if len(to_restore_dict) > 0:
            checkpoint = torch.load(self.restore_path)
            Checkpoint.load_objects(to_load=to_restore_dict, checkpoint=checkpoint)
            print('Restored all variables from {}'.format(self.restore_path))
        else:
            print('Clear start')

        to_save_dict = dict()
        if self.save_ckpt is True:
            if self.network is not None:
                to_save_dict.update({"model": self.network})
            if self.optimizer is not None:
                to_save_dict.update({"optimizer": self.optimizer})
            if self.lr_policy is not None:
                to_save_dict.update({"lr_scheduler": self.lr_policy})
        if self.save_session is True:
            to_save_dict.update({'engine': engine})
        if len(to_save_dict) > 0:
            def _final_func(engine):
                return engine.state.iteration
            self.final_checkpoint = Checkpoint(
                to_save=to_save_dict,
                score_function=_final_func,
                score_name="final_step",
                save_handler=DiskSaver(self.save_path, require_empty=False)
            )
            if self.save_val_model is True:
                def _score_func(engine):
                    return engine.state.metrics[self.key_metric_name]
                self.val_checkpoint = Checkpoint(
                    to_save=to_save_dict,
                    n_saved=self.val_n_saved,
                    score_function=_score_func,
                    score_name="key_metric",
                    save_handler=DiskSaver(self.save_path, require_empty=False)
                )
            if self.epoch_save_interval > 0:
                def _epoch_func(engine):
                    return engine.state.epoch
                self.epoch_checkpoint = Checkpoint(
                    to_save=to_save_dict,
                    n_saved=self.epoch_n_saved,
                    score_function=_epoch_func,
                    score_name="epoch",
                    save_handler=DiskSaver(self.save_path, require_empty=False)
                )
            if self.step_save_interval > 0:
                def _step_func(engine):
                    return engine.state.iteration
                self.step_checkpoint = Checkpoint(
                    to_save=to_save_dict,
                    n_saved=self.step_n_saved,
                    score_function=_step_func,
                    score_name="step",
                    save_handler=DiskSaver(self.save_path, require_empty=False)
                )

    def completed_callback(self, engine):
        if self.save_ckpt is True or self.save_session is True:
            self.final_checkpoint(engine)
            print('Train completed, saved final checkpoint: {}'.format(self.final_checkpoint.last_checkpoint))

    def epoch_completed_callback(self, engine):
        if self.save_ckpt is True or self.save_session is True:
            if self.epoch_save_interval > 0 and engine.state.epoch % self.epoch_save_interval == 0:
                self.epoch_checkpoint(engine)
                print('Saved epoch checkpoint: {}'.format(self.epoch_checkpoint.last_checkpoint))
            if self.save_val_model is True:
                self.val_checkpoint(engine)
                print('Saved val checkpoint: {}'.format(self.val_checkpoint.last_checkpoint))

    def step_completed_callback(self, engine):
        if self.save_ckpt is True or self.save_session is True:
            if self.step_save_interval > 0 and engine.state.iteration % self.step_save_interval == 0:
                self.step_checkpoint(engine)
                print('Saved step checkpoint: {}'.format(self.step_checkpoint.last_checkpoint))

    def exception_raised_callback(self, engine, e):
        if self.save_ckpt is True or self.save_session is True:
            self.final_checkpoint(engine)
            print('Exception_raised, saved exception checkpoint: {}'.format(self.final_checkpoint.last_checkpoint))
