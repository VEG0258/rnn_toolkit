import torch
import torch.nn as nn
from .loss import loss_function

#PyTorch enables gradient computation by default in training mode, so no need to explicitly add with torch.enable_grad()
def train(model, device, hidden, train_loader, normalization, optimizer, sequence_length, gradient_clipping):
    model.train() 
    total_loss = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = model.repackage_hidden(hidden)

        optimizer.zero_grad()
        outputs, hidden = model.forward(inputs, hidden) 
        loss = loss_function(outputs, labels, sequence_length, normalization)  
        loss.backward() 
        if gradient_clipping == True:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
        optimizer.step()

        total_loss += loss.item() #accumulate batch loss

    avg_loss = total_loss/len(train_loader)
    print(f"Train - Loss: {avg_loss}")

    return model, hidden, avg_loss


