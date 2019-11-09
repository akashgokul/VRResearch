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

# In[ ]:


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
class MTurkTrain(Dataset):
  def __init__(self,csv_file):
    self.data_frame = pd.read_csv(csv_file)
    self.img_dir = "/global/scratch/oafolabi/data/mturkCSVs/m-turk"

  def __len__(self):
    return self.data_frame.shape[0]

  def __getitem__(self,idx):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img_label_pair = self.data_frame.iloc[idx]
    img_name = img_label_pair[0]
    img = Image.open(self.img_dir +'/'+ img_name)
    img = transform(img)
    label = img_label_pair[1]
    return img,label

train_dataset = MTurkTrain("/global/scratch/oafolabi/data/mturkCSVs/train_data.csv")
train_size = train_dataset.__len__()

params_t = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 0}
training_generator = data.DataLoader(train_dataset, **params_t)


validation_set = MTurkTrain("val_data.csv")
validation_size = validation_set.__len__()
params_v = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 0}
validation_generator = data.DataLoader(validation_set, **params_v)


# **Training**

# In[ ]:



model = models.resnet18()
#From Stanford tutorial
#use_cuda = torch.cuda.is_available()
#device = torch.device("cuda:0" if use_cuda else "cpu")


max_epochs = 1

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters())

start_ts = time.time()
model.train()

for epoch in range(max_epochs):
    print("EPOCH: " + str(epoch))
    total_loss = 0
  #Training
    for idx, data in enumerate(training_generator):
        print(data.shape)
        X, y = data[0], data[1]
        model.zero_grad()
        outputs = model(X)
        print("     on to loss")
        loss = loss_function(outputs, y)
        loss.backward()
        print("     backprop")
        optimizer.step()
        current_loss = loss.item()
        total_loss += current_loss
        print("     Loss: {:.4f}".format(total_loss/(idx+1)))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()



# **Save Model**

# In[ ]:


#model_save_name = 'resnet18.pt'
#path = "{model_save_name}"
#torch.save(model.state_dict(), path)


# **Load Pre-saved Model**

# In[ ]:


#model_save_name = 'resnet18.pt'
#path = "{model_save_name}"
#model.load_state_dict(torch.load(path))


# **Validation**

# In[ ]:


# Validation

model.eval()
with torch.set_grad_enabled(False):
  val_wrong = 0
  for i, data in enumerate(validation_generator):
    print("Current: " + str(i) + " / " )
    # Transfer to GPU
    X, y = data[0], data[1]
     # Model computations
    outputs = model(X)
    predicted_classes = torch.max(outputs, 1)[1]
    prediction_lst = predicted_classes.tolist()
    val_wrong += sum([1 if prediction_lst[i] == y[i] else 0 for i in range(len(prediction_lst))])

#print(f"Training time: {time.time()-start_ts}s")


# In[ ]:


print(1 - (val_wrong / 387))
