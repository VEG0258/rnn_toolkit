import numpy as np
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, numstates, n_neurons, batch_size, hidden_sizes, dropout):  #default setting: recurrent activation = sigmoid function, activation function for the cell state = tanh, output activation = tanh 
        super(LSTMModel, self).__init__()
        self.batch_size = batch_size
        self.n_neurons = n_neurons
        self.hidden_sizes = hidden_sizes
        #Feed Forward Dense Layers
        self.feedforward_layers = self.create_feedforward_layers(numstates, hidden_sizes)
        #LSTM
        self.lstm = nn.LSTM(input_size=self.hidden_sizes[-1],
                            hidden_size=n_neurons,
                            batch_first=True)
        #Drop Out Layer
        self.drop1 = nn.Dropout(dropout)
        #Final Dense layer
        self.finaldense = nn.Linear(n_neurons, numstates)
        #Normalization Layers
        self.lstm_norm = nn.LayerNorm(n_neurons)
    
    #Create feed forward layers
    #if hidden_size is [20, 30, 40], then the model will create three feedforward layers with (input_dim, 20), (20, 30), (30, 40)
    def create_feedforward_layers(self, input_dim, hidden_sizes):
        layers = [] 
        if hidden_sizes is None:
            return None
        else:
            for i, hidden_size in hidden_sizes:
                if i == 0:
                    layers.append(nn.Linear(input_dim, hidden_size))
                else:
                    layers.append(nn.Linear(hidden_size[-1], hidden_size))
                layers.append(nn.ReLU())
            return nn.Sequential(*layers)
        

    #Initalizing the hidden states
    def zero_init_hidden(self):
        weight = next(self.parameters()).data
        hidden = (weight.new_zeros(1, self.batch_size, self.n_neurons),
                  weight.new_zeros(1, self.batch_size, self.n_neurons))
        return hidden

    #E.g., to create an initializer function that initializes the state with a mean of zero and standard deviation of 0.1, we call make_gaussian_state_initializer(zero_init_hidden, stddev=0.01). 
    def make_gaussian_state_initializer(self, initializer, noise=False, mean = 0, stddev=0.1):
        def gaussian_state_initializer(shape, batch_size, dtype, index):
            init_state = initializer(shape, batch_size, dtype, index)
            if noise == True:
                print("Noisy Initial States")
                print("mean: ", mean)
                print("stddev ", stddev)
                return lambda: init_state + torch.randn_like(init_state) * stddev + mean
            else:
                print("Zero Intial States")
                return lambda: init_state
            
    #Wraps hidden states in new Variables, to detach them from their history.       
    def repackage_hidden(self,h):
        if type(h) == tuple:
            return tuple(self.repackage_hidden(v) for v in h)
        else:
            return h.detach()

    def extract_hidden(self, hidden):
        if self.rnn_type == 'LSTM':
            return hidden[0][-1].data.cpu()  # hidden state last layer (hidden[1] is cell state)
        else:
            return hidden[-1].data.cpu()  # last layer

    
    def forward(self, x, hidden):
        #Feed Forward Dense Layer
        for layer in self.feedforward_layers:
            x = layer(x)

        #LSTM
        x, hidden = self.lstm(x, hidden) 
        #Layer Normalization after LSTM
        x = self.lstm_norm(x)
        x = self.drop1(x)

        #Final Dense Layers
        x = self.finaldense(x)
        return x, hidden
    
