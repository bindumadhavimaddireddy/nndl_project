#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision

from torch.utils.data import Dataset, DataLoader, BatchSampler, random_split
from torchvision import transforms
from PIL import Image


# In[10]:


# Create Dataset class for multilabel classification
class MultiClassImageDataset(Dataset):
    def __init__(self, ann_df, super_map_df, sub_map_df, img_dir, transform=None):
        self.ann_df = ann_df 
        self.super_map_df = super_map_df
        self.sub_map_df = sub_map_df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ann_df)

    def __getitem__(self, idx):
        img_name = self.ann_df['image'][idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        super_idx = self.ann_df['superclass_index'][idx]
        super_label = self.super_map_df['class'][super_idx]
        
        sub_idx = self.ann_df['subclass_index'][idx]
        sub_label = self.sub_map_df['class'][sub_idx]
        
        if self.transform:
            image = self.transform(image)  
            
        return image, super_idx, super_label, sub_idx, sub_label

class MultiClassImageTestDataset(Dataset):
    def __init__(self, super_map_df, sub_map_df, img_dir, transform=None):
        self.super_map_df = super_map_df
        self.sub_map_df = sub_map_df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self): # Count files in img_dir
        return len([fname for fname in os.listdir(self.img_dir)])

    def __getitem__(self, idx):
        img_name = str(idx) + '.jpg'
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)  
            
        return image, img_name


# In[11]:


# from torchvision import transforms

train_ann_df = pd.read_csv('train_data.csv')
# test_ann_df = pd.read_csv('test_data.csv')
super_map_df = pd.read_csv('superclass_mapping.csv')
sub_map_df = pd.read_csv('subclass_mapping.csv')

train_img_dir = 'train_images'
test_img_dir = 'test_images'


clip_transform_train = transforms.Compose([
    transforms.Resize((224, 224)),  # CLIP expects 224x224
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomRotation(degrees=30),
    transforms.RandomAffine(degrees=0, shear=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Test transform (no augmentation)
clip_transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create train and val split
train_dataset = MultiClassImageDataset(train_ann_df, super_map_df, sub_map_df, train_img_dir, transform=clip_transform_train)
train_dataset, val_dataset = random_split(train_dataset, [0.9, 0.1]) 

# Create test dataset
test_dataset = MultiClassImageTestDataset(super_map_df, sub_map_df, test_img_dir, transform=clip_transform_test)

# Create dataloaders
batch_size = 64
train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)

val_loader = DataLoader(val_dataset, 
                        batch_size=batch_size, 
                        shuffle=True)

test_loader = DataLoader(test_dataset, 
                         batch_size=1, 
                         shuffle=False)


# In[12]:


def detect_novel_class(logits, threshold=1.0, novel_index=3):
    energy = torch.logsumexp(logits, dim=1)
    _, predicted_labels = torch.max(logits, dim=1)
    predicted_labels[energy < threshold] = novel_index
    return predicted_labels


# In[13]:


import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor

class CLIPAdapterModel(nn.Module):
    def __init__(self, num_super=4, num_sub=88):
        super(CLIPAdapterModel, self).__init__()
        
        # Load CLIP vision encoder
        self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPImageProcessor()

        # Freeze all CLIP layers
        for param in self.clip.parameters():
            param.requires_grad = False

        # Adapter head
        clip_dim = self.clip.config.hidden_size

        self.super_head = nn.Sequential(
            nn.Linear(clip_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, num_super)
        )

        self.sub_head = nn.Sequential(
            nn.Linear(clip_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, num_sub)
        )

    def forward(self, pixel_values):
        outputs = self.clip(pixel_values=pixel_values)
        features = outputs.last_hidden_state[:, 0, :]  # Take [CLS] token

        super_logits = self.super_head(features)
        sub_logits = self.sub_head(features)

        return super_logits, sub_logits


# In[14]:


class CLIPTrainer:
    def __init__(self, model, criterion, optimizer, train_loader, val_loader, test_loader=None, device='cuda'):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
            
        for i, data in enumerate(self.train_loader):
            images, super_labels, sub_labels = data[0].to(self.device), data[1].to(self.device), data[3].to(self.device)

            self.optimizer.zero_grad()
            super_outputs, sub_outputs = self.model(images)
            loss = self.criterion(super_outputs, super_labels) + self.criterion(sub_outputs, sub_labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        print(f"Train Loss: {running_loss / len(self.train_loader):.4f}")

    def validate_epoch(self, energy_threshold_super=1.5, energy_threshold_sub=2.0):
        # self.model.eval()
        super_correct = 0
        sub_correct = 0
        total = 0
        running_loss = 0.0

        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                images, super_labels, sub_labels = data[0].to(self.device), data[1].to(self.device), data[3].to(self.device)

                super_outputs, sub_outputs = self.model(images)

                loss = self.criterion(super_outputs, super_labels) + self.criterion(sub_outputs, sub_labels)
                super_preds = detect_novel_class(super_outputs, energy_threshold_super, novel_index=3)
                sub_preds = detect_novel_class(sub_outputs, energy_threshold_sub, novel_index=87)

                total += super_labels.size(0)
                super_correct += (super_preds == super_labels).sum().item()
                sub_correct += (sub_preds == sub_labels).sum().item()
                running_loss += loss.item()  

        print(f'Validation loss: {running_loss/i:.3f}')
        print(f"Val Superclass Acc: {100 * super_correct / total:.2f}%")
        print(f"Val Subclass Acc: {100 * sub_correct / total:.2f}%")

    def test(self, save_to_csv=True, return_predictions=False):
        # self.model.eval()
        if not self.test_loader:
            raise NotImplementedError('test_loader not specified')

        # Evaluate on test set, in this simple demo no special care is taken for novel/unseen classes
        test_predictions = {'image': [], 'superclass_index': [], 'subclass_index': []}

        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                images, img_names = data[0].to(self.device), data[1]
                super_outputs, sub_outputs = self.model(images)

                super_preds = detect_novel_class(super_outputs, threshold=1.5, novel_index=3)
                sub_preds = detect_novel_class(sub_outputs, threshold=2.0, novel_index=87)

                test_predictions['image'].append(img_names[0])
                test_predictions['superclass_index'].append(super_preds.item())
                test_predictions['subclass_index'].append(sub_preds.item())

        test_predictions = pd.DataFrame(data=test_predictions)
        if save_to_csv:
            test_predictions.to_csv('test_predictions.csv', index=False)
        if return_predictions:
            return test_predictions


# In[15]:


import argparse

# Init model and trainer
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cpu', help="Device to run on: 'cpu' or 'cuda'")
args = parser.parse_args()
device = args.device

print("----device------", device)

# device = 'cpu'
model = CLIPAdapterModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
trainer = CLIPTrainer(model, criterion, optimizer, train_loader, val_loader, test_loader, device=device)


# In[16]:


# Training loop
for epoch in range(10):
    print(f'Epoch {epoch+1}')
    trainer.train_epoch()
    trainer.validate_epoch(energy_threshold_super=1.5, energy_threshold_sub=2.0)

    print('')

print('Finished Training')


# In[17]:


test_predictions = trainer.test(save_to_csv=True, return_predictions=True)
print("\nTest predictions saved to test_predictions.csv")

