import pandas as pd
import torch
from PIL import Image
from torch.utils import data
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time
from torch import nn, optim
import torchvision.models as models

# **Load Data**

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class MTurkTrain(Dataset):
  def __init__(self,csv_file):
    self.data_frame = pd.read_csv(csv_file)
    self.img_dir = "/global/scratch/oafolabi/data/mturkCSVs/m-turk"

  def __len__(self):
    return self.data_frame.shape[0]

  def __getitem__(self,idx):
    transform = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img_label_pair = self.data_frame.iloc[idx]
    img_name = img_label_pair[0]
    img = Image.open(self.img_dir +'/'+ img_name)
    img = transform(img)
    label = img_label_pair[1]
    return img,label


## MODEL NAME, LOAD!!! ##
model = models.resnet152()

model.load_state_dict(torch.load("fcnresnet101.pt",map_location=torch.device('cpu')))

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)

# # **Validation**
#
# # In[ ]:


# Validation

validation_set = MTurkTrain("/global/scratch/oafolabi/data/mturkCSVs/val_data.csv")
validation_size = validation_set.__len__()
params_v = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 0}
validation_generator = data.DataLoader(validation_set, **params_v)

print("--------------")
print("Starting Validation Testing:")
model.eval()
with torch.set_grad_enabled(False):
  val_wrong = 0
  total = 0
  for i, data in enumerate(validation_generator):
    # Transfer to GPU
    X, y = data[0].to(device), data[1].to(device)
    y = y.item()
    outputs = model(X)
    predicted_class = torch.argmax(outputs)
    prediction = predicted_class.item()
    val_wrong += sum([1 if prediction != y else 0])



# In[ ]:

print("Validation Accuracy: ")
val_acc = 1 - (val_wrong / 387)
print(val_acc)
# if val_acc >= 0.7:
#     PATH = 'fcnresnet101.pt'
#     torch.save(model.state_dict(), PATH)
validation_set = MTurkTrain("full_gb.csv")
