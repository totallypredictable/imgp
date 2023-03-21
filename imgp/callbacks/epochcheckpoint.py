from tensorflow.keras.callbacks import Callback
import os


class EpochCheckpoint(Callback):
    def __init__(self, output_path: str, every: int = 5, start_at: int = 0):
        super().__init__()

        # store the base output path for the model, the number of epochs
        # that must pass before the model is serialised to disk and the current
        # epoch value
        self.output_path = output_path
        self.every = every
        self.int_epoch = start_at

    def on_epoch_end(self, epoch, logs={}):
        # check to see if the model should be serialised to disk
        if (self.int_epoch + 1) % self.every == 0:
            p = os.path.sep.join(
                [self.output_path, "epoch_{}.hdf5".format(self.int_epoch + 1)]
            )
            self.model.save(p, overwrite=True)

        # increment the internal epoch counter
        self.int_epoch += 1
