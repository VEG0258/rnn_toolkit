import numpy as np
import torch
import torch.nn as nn

#The output should has a shape of (batch_size * num_class * sequence_length) NO SOFTMAX
#The targets should has a shape of (batch_size * sequence_length) NOT ONE HOT ENCODED
def loss_function(outputs, targets, sequence_length, normalization):
    if normalization:
        reduction = 'none'
    else:
        reduction = 'mean'

    criterion = nn.CrossEntropyLoss(reduction=reduction)

    if normalization == True:
        #reshape outputs
        outputs = outputs.permute(0, 2, 1)
        #get raw_loss
        raw_loss = criterion(outputs, targets)  # (batch_size, seq_length)
        #sum the loss terms along the sequence
        summed_loss = torch.sum(raw_loss, dim=1)  # (batch_size,)
        #normalize
        normalized_loss = summed_loss / sequence_length
        #averaged across the batch
        final_loss = torch.mean(normalized_loss) 
    else:
        final_loss = criterion(outputs, targets)
    
    return final_loss