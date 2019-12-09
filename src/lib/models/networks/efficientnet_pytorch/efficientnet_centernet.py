from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet

class EfficientNetCenternet(nn.Module):
	def __init__():
		self.efficientnet=EfficientNet.from_name('efficientnet-b0')
		pass
	def forward(x):
		y=[]
		result=self.efficientnet(x)
