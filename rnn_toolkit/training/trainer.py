import torch
import torch.nn as nn

def train_model(model, train_loader, loss, optimizer, batch_size, sequence_length, gradient_clipping, num_epochs=1000):
    model.train() 
    hidden = model.make_gaussian_state_initializer(model.zero_init_hidden, noise = True)

    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = model.repackage_hidden(hidden)

            optimizer.zero_grad()
            outputs, hidden = model.forward(inputs, hidden) 
            loss = loss(outputs, labels, sequence_length)  
            loss.backward() 
            if gradient_clipping == True:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}, Last norm: {loss}")

