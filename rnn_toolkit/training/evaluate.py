import torch
import torch.nn as nn
from .loss import loss_function

def evaluate(model, criterion, hidden, test_loader, sequence_length):
    model.eval()
    total_loss = 0

    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in test_loader:
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = model.repackage_hidden(hidden)

            outputs, hidden = model.forward(inputs, hidden) 
            loss =loss_function(criterion, outputs, labels, sequence_length, normalization)  
            total_loss += loss.item()
        
    avg_loss = total_loss / len(test_loader)
    print(f"Evaluation - Loss: {avg_loss}")

    return avg_loss

