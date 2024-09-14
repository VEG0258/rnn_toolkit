import matplotlib.pyplot as plt
import numpy as np

def loss_plot(train_loss, test_loss, time_steps):
    plt.figure(figsize = (10, 10))

    # Plot train and test loss
    plt.plot(train_loss[:time_steps], label = "Train loss")
    plt.plot(test_loss[:time_steps], label = "Test loss")
    plt.title("LSTM loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")

    plt.savefig(f"tests/loss_plot.png")
    plt.close()
    print("loss plot printed")