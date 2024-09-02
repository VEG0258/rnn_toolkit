import torch
import torch.nn as nn

def train_model(model, train_loader, loss, optimizer, batch_size, sequence_length, gradient_clipping, num_epochs=1000):
    model.train() 
    hidden_states_h = []
    hidden_states_c = []

    for epoch in range(num_epochs):
        hidden = model.make_gaussian_state_initializer(model.zero_init_hidden, noise = True)
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

        h, c = hidden
        hidden_states_h.append(h)
        hidden_states_c.append(c)

        #Save the model
        save_frequency = 5
        if (epoch + 1) % save_frequency == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'hidden_states_h': hidden_states_h,
                'hidden_states_c': hidden_states_c
            }, f"LSTM/model/epoch_{epoch + 1}.pt")
            print(f"Model checkpoint saved at epoch {epoch + 1}")


        print(f"Epoch {epoch+1}, Loss: {loss.item()}, Last norm: {loss}")

    return model, hidden_states_h, hidden_states_c


