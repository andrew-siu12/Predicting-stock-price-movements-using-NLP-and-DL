import os
from base.base_train import BaseTrain
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt

class ModelTrainer(BaseTrain):
    def __init__(self, model, data, config):
        super(ModelTrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.f1 = []
        self.val_loss = []
        self.val_acc = []
        self.val_f1 = []
        self.init_saver()

    def init_saver(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.checkpoint_dir,
                                      '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp_name),
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                save_weights_only=True,
                verbose=True
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.summary_dir,
                write_graph=True,
            )
        )

        self.callbacks.append(
            EarlyStopping(
                patience=5,
                restore_best_weights=True
            )
        )

    def train(self):
        history = self.model.fit(
            [self.data[2], self.data[0]], self.data[1],
            epochs=self.config.num_epochs,
            batch_size = self.config.batch_size,
            verbose=2,
            validation_split=self.config.validation_split,
            callbacks=self.callbacks
        )
        self.loss.extend(history.history["loss"])
        self.acc.extend(history.history["accuracy"])
        self.f1.extend(history.history["f1"])
        self.val_loss.extend(history.history["val_loss"])
        self.val_acc.extend(history.history["val_accuracy"])
        self.val_f1.extend(history.history["val_f1"])
        self.epoch = history.epoch

    def visualize(self, title):
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        plt.style.use("ggplot")
        ax[0].plot(self.epoch, self.loss, label="train_loss")
        ax[0].plot(self.epoch, self.val_loss, label="val_loss")
        ax[1].plot(self.epoch, self.acc, label="train_acc")
        ax[1].plot(self.epoch, self.val_acc, label="val_acc")

        for i in range(2):
            ax[i].legend()
            ax[i].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[1].set_ylabel("Accuracy")
        ax[0].set_title(f"Training and Validation loss {title}")
        ax[1].set_title(f"Training and Validation Accuracy {title}")
        plt.show()