import torch
import numpy as np


def trainning_phase(model, data_loader, optimizer, scheduler, loss_fn, device):

    model.train()
    scheduler.step()

    total_loss = 0.0
    total_correct_predictions = 0
    total_samples = 0

    for iteration in range(1, len(data_loader) + 1):

        input1, input2, labels = next(iter(data_loader))

        input1 = torch.autograd.Variable(input1, requires_grad=True).to(device)
        input2 = torch.autograd.Variable(input2, requires_grad=True).to(device)
        labels = torch.autograd.Variable(labels).to(device)

        optimizer.zero_grad()

        outputs = model(input1, input2)
        probs = torch.nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct_predictions += (preds == labels).sum().item()
        total_samples += input1.size(0)

        print(f"Iteration [{iteration}/{len(data_loader)}] - [Train]: LOSS = {round(loss.item(), 4)} || ACCURACY = {round((preds == labels).sum().item() / input1.size(0), 4)}")

    epoch_loss = total_loss / len(data_loader)
    epoch_accuracy = (total_correct_predictions / total_samples)

    return epoch_loss, epoch_accuracy

def testing_phase(model, data_loader, loss_fn, device):

    model.eval()
    total_loss = 0.0
    total_correct_predictions = 0
    total_samples = 0

    for iteration in range(1, len(data_loader) + 1):

        input1, input2, labels = next(iter(data_loader))

        input1 = torch.autograd.Variable(input1, requires_grad=True).to(device)
        input2 = torch.autograd.Variable(input2, requires_grad=True).to(device)
        labels = torch.autograd.Variable(labels).to(device)


        outputs = model(input1, input2)
        probs = torch.nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]
        loss = loss_fn(outputs, labels)

        total_loss += loss.item()
        total_correct_predictions += (preds == labels).sum().item()
        total_samples += input1.size(0)

        print(f"Iteration [{iteration}/{len(data_loader)}] - [Train]: LOSS = {round(loss.item(), 4)} || ACCURACY = {round((preds == labels).sum().item() / input1.size(0), 4)}")

    epoch_loss = total_loss / len(data_loader)
    epoch_accuracy = (total_correct_predictions / total_samples)

    return epoch_loss, epoch_accuracy


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

class EarlyStopping:
    def __init__(self, patience:int=10, delta=0, verbose=False, path:str='checkpoint.pth'):

        """
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            delta (float): Minimum change in monitored quantity to qualify as an improvement.
            verbose (bool): If True, it prints a message for each improvement.
            path (str): Path for the checkpoint to be saved.
        """

        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
