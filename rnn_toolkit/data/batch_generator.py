import numpy as np
import torch
import torch.nn.functional as F

#Help Method1
def get_batches(data,batch_size,sequence_length,lag=1):

    num_batch = int(np.size(data,1)/(sequence_length))
    X=np.zeros((batch_size*num_batch,sequence_length))
    Y=np.zeros((batch_size*num_batch,sequence_length))
    c=0
    
    for f in range(batch_size):
        c=f
        for i in range(num_batch):
            m=i*sequence_length
            if m+sequence_length+lag<data.shape[1]:
                X[c,:]=data[f,m:m+sequence_length]
                Y[c,:]=data[f,m+lag:m+sequence_length+lag]
                c+=batch_size
            
    return [X,Y]

#Option on if what to perfome one-hot encoding on target
#NOTES: Crossentropy do not accept target data in one-hot encoding format, it accepts shape (N, d1, d2, ..., dK)
def batch_data(dt, batch_size, sequence_length, numstates, lag, v, one_hot_target):
    inp,targ=get_batches(dt,batch_size,sequence_length,lag=lag)

    #breaking the data into training and test sets
    input_train=inp[:int(v*inp.shape[0]),:]
    input_test=inp[int(v*inp.shape[0]):,:]
    target_train=targ[:int(v*inp.shape[0]),:]
    target_test=targ[int(v*inp.shape[0]):,:]

    #one hot encoding on input only (crossentropy do not accept target data in one-hot encoding format)
    input_train = F.one_hot(input_train, num_classes=numstates)
    input_test = F.one_hot(input_test, num_classes=numstates)
    if one_hot_target is True:
        target_train = F.one_hot(target_train, num_classes=numstates)
        target_test = F.one_hot(target_test, num_classes=numstates)

    #to tensor
    input_train = torch.tensor(input_train, dtype=torch.float32)
    input_test = torch.tensor(input_test, dtype=torch.float32)
    target_train = torch.tensor(target_train, dtype=torch.float32)
    target_test = torch.tensor(target_test, dtype=torch.float32)

    return input_train, input_test, target_train, target_test