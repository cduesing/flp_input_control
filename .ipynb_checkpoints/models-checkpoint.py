from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F


# a toy model used for test purposes only
class ANNetwork(nn.Module):

    # init the model from predefined aarchitecture
    def __init__(self, num_features, num_classes):

        super().__init__()

        self.num_classes = num_classes
        self.fc1 = nn.Linear(num_features, 50)
        self.fc2 = nn.Linear(50, 30)
        self.fc3 = nn.Linear(30, 15)
        self.fc4 = nn.Linear(15, num_classes)
                    
    # forward pass
    # cross-entropy loss
    # softmax activation
    def forward(self,x,labels=None):

        logits = F.relu(self.fc1(x))
        logits = F.relu(self.fc2(logits))
        logits = F.relu(self.fc3(logits))
        logits = self.fc4(logits)
        logits = torch.sigmoid(logits)
        loss = None
        if labels is not None:
            assert len(x) == len(labels), "x and y have to be of same length"
            
            if self.num_classes > 1:
                loss_fct = get_proper_loss_function(labels.dtype)
            else: 
                loss_fct = nn.MSELoss()
                labels = labels.float()

            loss = loss_fct(logits, labels)
        return (loss, logits)


# takes label dtype
# returns proper loss function to use
def get_proper_loss_function(data_type):
    if data_type == torch.long or data_type == torch.int:
        return nn.CrossEntropyLoss()
    else:
        return nn.BCEWithLogitsLoss()

    
# takes a string as input
# returns a pytorch model w.r.t. the provided name
def get_model_by_name(config):
    if config["model_name"].lower() == "ann":
        return ANNetwork(config["num_features"], config["num_classes"])
    else:
        raise ValueError("model " + config["model_name"] + " has not been configured yet")