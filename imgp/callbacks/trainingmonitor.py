from tensorflow.keras.callbacks import BaseLogger, Callback
import matplotlib.pyplot as plt
import numpy as np
import json
import os


class TrainingMonitor(BaseLogger):
    def __init__(self, fig_path: str, json_path: str | None = None, start_at: int = 0):
        # store the output path for the figure, the path to the JSON serialised
        # file, and the starting epoch
        super().__init__()
        self.fig_path = fig_path
        self.json_path = json_path
        self.start_at = start_at

    def on_train_begin(self, logs: dict = {}) -> None:
        # initialise the histry dictionary
        self.H = {}

        # if the JSON history path exists, load the training history
        if self.json_path is not None:
            if os.path.exists(self.json_path):
                self.H = json.loads(open(self.json_path).read())

                # check to see if a starting epoch was supplied
                if self.start_at > 0:
                    # loop over the entries in the history log and trim any
                    # entries that are past the starting epoch
                    for k in self.H.keys():
                        self.H[k] = self.H[k][: self.start_at]

    def on_epoch_end(self, epoch: int, logs: dict = {}):
        # loop over the logs and update the loss, accuracy etc. for the
        # entire training process
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l

        if self.json_path is not None:
            f = open(self.json_path, "w")
            f.write(json.dumps(self.H))
            f.close()

        # ensure at least two epochs have passed before plotting
        # (epoch starts at zero)
        if len(self.H["loss"]) > 1:
            # plot the training loss and accuracy
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["loss"], label="training_loss")
            plt.plot(N, self.H["val_loss"], label="val_loss")
            plt.plot(N, self.H["accuracy"], label="train_acc")
            plt.plot(N, self.H["val_accuracy"], label="val_accuracy")
            plt.title(
                "Training Loss and Accuracy [Epoch{}]".format(len(self.H["loss"]))
            )
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()

            # save the figure
            plt.savefig(self.fig_path)
            plt.close()
