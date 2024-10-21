# Description: Contains utility functions for training
import pickle

import matplotlib.pyplot as plt
import torch


def save_results(
        model,
        model_path,
        training_loss_logger,
        validation_loss_logger,
        training_accuracy_logger,
        validation_accuracy_logger,
):
    # Save the model
    torch.save(model.state_dict(), model_path)
    # Save the training logs
    with open("training_loss_logger.pkl", "wb") as f:
        pickle.dump(training_loss_logger, f)
    with open("validation_loss_logger.pkl", "wb") as f:
        pickle.dump(validation_loss_logger, f)
    with open("training_accuracy_logger.pkl", "wb") as f:
        pickle.dump(training_accuracy_logger, f)
    with open("validation_accuracy_logger.pkl", "wb") as f:
        pickle.dump(validation_accuracy_logger, f)


def visualize_loss(training_loss_logger, validation_loss_logger, path):
    plt.plot(training_loss_logger, label="Training Loss")
    plt.plot(validation_loss_logger, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Training Loss", "Validation Loss"], loc="upper left")
    # Save the plot
    plt.savefig(path)
    # clear the plot
    plt.clf()


def visualize_accuracy(training_accuracy_logger, validation_accuracy_logger, path):
    plt.plot(training_accuracy_logger, label="Training Accuracy")
    plt.plot(validation_accuracy_logger, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Training Accuracy", "Validation Accuracy"], loc="upper left")
    # Save the plot
    plt.savefig(path)
    # clear the plot
    plt.clf()
