from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as save_image
import torchvision.models as models
import os.path

# Select this features as in the paper
vgg_feature_layers = [0, 5, 10, 19, 28]

class VGG(nn.Module):
    def __init__(self, feature_layers, model_path=None):
        super(VGG, self).__init__()
        self.feature_layers = feature_layers
        # After the last we want all are unecessary
        
        if model_path is not None and os.path.isfile(model_path):
            model_pre = torch.load(model_path)
        else:
            model_pre = models.vgg19(pretrained=True)
        self.model = model_pre.features[:self.feature_layers[-1]+1]

    def forward(self, x):
        feature_list = []
        for n_layer, layer in enumerate(self.model):
            # Pass the input through each layer and check if is one of interest
            x = layer(x)
            if n_layer in self.feature_layers:
                feature_list.append(x)
        return feature_list