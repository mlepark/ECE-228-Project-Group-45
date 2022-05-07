## Defining the linear Convolutional Neural Network model here ##

import torch
import torch.nn as nn

cfg = {'cnn_test': ['F', 64, 'M', 128, 128, 'M4', 256, 256, 256, 'M4', 512, 512, 512, 'M4', 512, 512, 512, 'M']}

class CNN(nn.Module):

  def __init__(self, ntwrk_name):
      super().__init__()
      # layer definitions start here
      self.features = self._make_layer(cfg[ntwrk_name])
      self.classifier = nn.Linear(512,3)

  def _make_layer(self, cfg):
    layers = []
    in_channels = 1
    for x in cfg:
      if x == 'M':
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

      elif x == 'M4':
        layers += [nn.MaxPool2d(kernel_size=4, stride=4)]

      elif x == 'F':  # This is for the 1st layer
        layers += [ nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)]
        in_channels = 64

      else:
        layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                   nn.BatchNorm2d(x),
                   nn.ReLU(inplace=True)]
        in_channels = x
    
    layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
    return nn.Sequential(*layers)


  def forward(self,x):
    out = self.features(x)
    out = out.view(out.size(0), -1)
    out = self.classifier(out)
    return out





def CNN_test(mdl_name, **kwargs):
  model = CNN(ntwrk_name = mdl_name, **kwargs)
  return model
