from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights
from torch.nn import Linear

def replace_layer(model):
    input_dim = model.fc.in_features
    new_fc_layer = Linear(input_dim,4)
    model.fc = new_fc_layer

def base_model_resnet18():
    model = resnet18(weights = ResNet18_Weights)
    #replacing fc layer in resnet
    replace_layer(model)
    return model

def base_model_resnet34():
    model = resnet34(weights = ResNet34_Weights)
    #replacing fc layer in resnet
    replace_layer(model)
    return model
