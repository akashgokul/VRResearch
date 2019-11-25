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
#
#
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")

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

train_dataset = MTurkTrain("/global/scratch/oafolabi/data/mturkCSVs/train_data.csv")
train_size = train_dataset.__len__()

params_t = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 0}
training_generator = data.DataLoader(train_dataset, **params_t)



# **Training**

# In[ ]:



model = models.resnet152()

# if torch.cuda.device_count() > 1:
#   print("Let's use", torch.cuda.device_count(), "GPUs!")
#   # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#   model = nn.DataParallel(model)
#
# model = model.to(device)

# max_epochs = 17
#
# loss_function = nn.CrossEntropyLoss()
# optimizer = optim.Adadelta(model.parameters())
#
# start_ts = time.time()
# model.train()
#
# loss_epoch_dict = {i:[] for i in range(max_epochs)}
#
# for epoch in range(max_epochs):
#     print("EPOCH: " + str(epoch))
#     total_loss = 0
#   #Training
#     for idx, data in enumerate(training_generator):
#         X, y = data[0].to(device), data[1].to(device)
#         print(y)
#         model.zero_grad()
#         outputs = model(X)
#         print(outputs)
#         loss = loss_function(outputs, y)
#         loss.backward()
#         optimizer.step()
#         current_loss = loss.item()
#         total_loss += current_loss
#         loss_epoch_dict[epoch].append(total_loss/(idx+1))
#         if(idx % 20 == 0):
#             print("     Loss: {:.4f}".format(total_loss/(idx+1)) + " EPOCH: " + str(epoch))
#         else:
#             print("     Loss: {:.4f}".format(total_loss/(idx+1)))
#
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#
#
#
# # **Save Model**
#
# # In[ ]:
#
# # **Load Pre-saved Model**
#
# # In[ ]:
#
#
# #model_save_name = 'resnet18.pt'
# #path = "{model_save_name}"
model.load_state_dict(torch.load("resnet152.pt",map_location=torch.device('cpu')))
#
#
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
    X, y = data[0], data[1]
    y = y.item()
     # Model computations
    outputs = model(X)
    predicted_class = torch.argmax(outputs)
    prediction = predicted_class.item()
    val_wrong += sum([1 if prediction != y else 0])
    total = i
print(total)
#print(f"Training time: {time.time()-start_ts}s")


# In[ ]:

print("Validation Accuracy (RESNET 152): ")
val_acc = 1 - (val_wrong / total)
print(val_acc)
# if val_acc >= 0.7:
#     PATH = 'fcnresnet101.pt'
#     torch.save(model.state_dict(), PATH)
