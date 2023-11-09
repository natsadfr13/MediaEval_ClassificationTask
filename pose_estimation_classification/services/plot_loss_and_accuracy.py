import matplotlib.pyplot as plt

def plotLossAndAccuracy(history):

        fig, axs = plt.subplots(1, 2, figsize = (15, 3))

        axs[0].plot(history.history['loss'], label='Train')
        axs[0].plot(history.history['val_loss'], label='Validation')
        axs[0].set_title('Loss')
        axs[0].set(ylabel='Loss')
        axs[0].legend()

        axs[1].plot(history.history['accuracy'], label='Train')
        axs[1].plot(history.history['val_accuracy'], label='Validation')
        axs[1].set_title('Accuracy')
        axs[1].set(ylabel='Accuracy')
        axs[1].legend()

        for ax in axs.flat:
            ax.set(xlabel='Epoch')

        plt.show()