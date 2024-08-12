import os
from collections import Counter
from functools import partial
from pathlib import Path
import tempfile

from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor

from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
import torch.distributed.checkpoint as dcp
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle

from model import *

def load_data(batch_size):

    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    #print(len(training_data))
    #print(Counter(training_data.class_to_idx))
    #print(training_data.targets.unique(return_counts=True))
    #print(len(test_dataloader))


    train_size = int(0.8 * len(training_data))
    validation_size = len(training_data) - train_size
    train_data, validation_data = random_split(training_data, [train_size, validation_size])
 
    #print(dict(Counter([label for _, label in train_data])))
    #print(dict(Counter([label for _, label in validation_data])))
  
    #default prefetch_factor=2
    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=8, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size, num_workers=8, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=8, shuffle=True)


    for X, y in train_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    return train_dataloader, validation_dataloader, test_dataloader

def train_model(config, device=None, model = None):

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = os.path.join(checkpoint_dir, "data.pkl")
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            model.load_state_dict(checkpoint_state["model_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
     
    else:
        start_epoch = 0

    trainloader, valloader, _ = load_data(batch_size=config["batch_size"])


    for epoch in range(start_epoch,10):  # loop over the dataset multiple times
        print(f"Epoch {epoch+1}\n-------------------------------")
        running_loss = 0.0
        epoch_steps = 0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Compute prediction error
            pred = model(inputs)
            loss = loss_fn(pred, labels)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for _, (inputs, labels) in enumerate(valloader):
            with torch.no_grad():
                inputs, labels = inputs.to(device), labels.to(device)

                pred = model(inputs)
                _, predicted = torch.max(pred.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = loss_fn(pred, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = os.path.join(checkpoint_dir,"data.pkl")
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report(
                {"loss": val_loss / val_steps, "accuracy": correct / total},
                checkpoint=checkpoint,
            )

    print("Finished Training")

    #return model, 

def test_accuracy(model, device):
    _, _, testloader = load_data(batch_size=64)

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            pred = model(inputs)
            _, predicted = torch.max(pred.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

def save_model(model, optimizer):

    print(f"Saving final model...")
    torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
                }, 'outputs/final_model.pth')

def main(max_num_epochs, num_samples):
    data_dir = os.path.abspath("./data")
    
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)

    print('Most kezdődik')
    #load_data()
    config = {
        "momentum": tune.choice([0.5, 0.9, 0.98]),
        "lr": tune.loguniform(0.0001, 0.1),
        "batch_size": tune.grid_search([32, 64, 128]),
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
    )
    result = tune.run(
        partial(train_model, device=device, model=model),
        resources_per_trial={"cpu": 2},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="loss", mode="min")
    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = os.path.join(checkpoint_dir, "data.pkl")
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

        model.load_state_dict(best_checkpoint_data["model_state_dict"])
        #save_model(model.load_state_dict(best_checkpoint_data["model_state_dict"]), model.load_state_dict(best_checkpoint_data["optimizer_state_dict"]))
        test_acc = test_accuracy(model, device)
        print("Best trial test set accuracy: {}".format(test_acc))

        try:
            os.makedirs(os.path.abspath("./outputs"))
        except FileExistsError:
            pass

        PATH = "./outputs/final_model.pth"
        torch.save(model.state_dict(), PATH)


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(max_num_epochs=10, num_samples=3)
