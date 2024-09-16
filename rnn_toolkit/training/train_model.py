import torch
import torch.nn as nn
from .train import train
from .evaluate import evaluate

def train_model(model, train_loader, test_loader, optimizer, batch_size, sequence_length, criterion, gradient_clipping, normalization, num_epochs=1000):
    model.eval() 
    hidden_states_h = []
    hidden_states_c = []
    train_loss = []
    test_loss = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}")

        hidden = model.make_gaussian_state_initializer(model.zero_init_hidden, noise = True)
        #train
        model, hidden, train_avg_loss= train(model, hidden, train_loader, criterion, normalization, optimizer, sequence_length, gradient_clipping, hidden_states_h, hidden_states_c)
        #storing train result
        train_loss.append(train_avg_loss)
        h, c = hidden
        hidden_states_h.append(h)
        hidden_states_c.append(c)

        #test
        test_avg_loss = evaluate(model, criterion, hidden, test_loader, sequence_length)
        #storing test result
        test_loss.append(test_avg_loss)

        #Save the model
        save_frequency = 5
        if (epoch + 1) % save_frequency == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_avg_loss,
                'test_loss': test_avg_loss,
                'hidden_states_h': hidden_states_h,
                'hidden_states_c': hidden_states_c
            }, f"LSTM/model/epoch_{epoch + 1}.pt")
            print(f"Model checkpoint saved at epoch {epoch + 1}")


    return model, hidden_states_h, hidden_states_c, train_loss, test_loss


