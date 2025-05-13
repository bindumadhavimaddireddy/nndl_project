#!/usr/bin/env python
# coding: utf-8

# In[105]:


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


# In[106]:


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
        ann_row = self.ann_df.iloc[idx]  # Safe positional access
        # img_name = self.ann_df['image'][idx]
        img_name = ann_row['image']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        super_idx = ann_row['superclass_index']
        super_label = self.super_map_df['class'][super_idx]
        
        sub_idx = ann_row['subclass_index']
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


# In[ ]:





# In[107]:


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

# Hold out 'reptile' superclass and one rare subclass as pseudo-novel
pseudo_novel_superclass = 2  # reptile
pseudo_novel_subclass = 86   # e.g., rare subclass

train_mask = ~((train_ann_df['superclass_index'] == pseudo_novel_superclass) |
               (train_ann_df['subclass_index'] == pseudo_novel_subclass))

pseudo_val_mask = ((train_ann_df['superclass_index'] == pseudo_novel_superclass) |
                   (train_ann_df['subclass_index'] == pseudo_novel_subclass))

# train_subset_df = train_ann_df[train_mask]
# val_subset_df = train_ann_df[pseudo_val_mask]

train_subset_df = train_ann_df[train_mask].reset_index(drop=True)
val_subset_df = train_ann_df[pseudo_val_mask].reset_index(drop=True)

# Rebuild datasets
train_dataset = MultiClassImageDataset(train_subset_df, super_map_df, sub_map_df, train_img_dir, transform=clip_transform_train)
val_dataset = MultiClassImageDataset(val_subset_df, super_map_df, sub_map_df, train_img_dir, transform=clip_transform_test)

# Recreate dataloaders
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# # Create train and val split
# train_dataset = MultiClassImageDataset(train_ann_df, super_map_df, sub_map_df, train_img_dir, transform=clip_transform_train)
# train_dataset, val_dataset = random_split(train_dataset, [0.9, 0.1]) 

# Create test dataset
test_dataset = MultiClassImageTestDataset(super_map_df, sub_map_df, test_img_dir, transform=clip_transform_test)

# Create dataloaders
batch_size = 64
train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)

val_loader = DataLoader(val_dataset, 
                        batch_size=batch_size, 
                        shuffle=False)

test_loader = DataLoader(test_dataset, 
                         batch_size=1, 
                         shuffle=False)


# In[108]:


def detect_novel_class(logits, threshold=1.0, novel_index=3):
    energy = torch.logsumexp(logits, dim=1)
    _, predicted_labels = torch.max(logits, dim=1)
    predicted_labels[energy < threshold] = novel_index
    return predicted_labels


# In[109]:


def tune_energy_threshold(model, val_loader, min_t=0.5, max_t=3.0, step=0.1, device="cpu"):
    best_threshold_super = 1.5
    best_threshold_sub = 2.0
    best_unseen_acc = 0.0

    # Simulate unseen samples in validation set
    for t_super in np.arange(min_t, max_t, step):
        for t_sub in np.arange(min_t, max_t, step):
            unseen_correct = 0
            unseen_total = 0

            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    images, super_labels, sub_labels, = data[0].to(device), data[1].to(device), data[3].to(device)

                    super_logits, sub_logits = model(images)

                    super_preds = detect_novel_class(super_logits, threshold=t_super, novel_index=3)
                    sub_preds = detect_novel_class(sub_logits, threshold=t_sub, novel_index=87)

                    unseen_mask = super_labels == 3  # ground truth novel superclass
                    unseen_correct += (super_preds[unseen_mask] == super_labels[unseen_mask]).sum().item()
                    unseen_total += unseen_mask.sum().item()

            if unseen_total > 0:
                acc = 100 * unseen_correct / unseen_total
                if acc > best_unseen_acc:
                    best_unseen_acc = acc
                    best_threshold_super = t_super
                    best_threshold_sub = t_sub

    print(f"Best thresholds found: super={best_threshold_super:.2f}, sub={best_threshold_sub:.2f}")
    return best_threshold_super, best_threshold_sub


# In[110]:


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


# In[111]:


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
        self.model.eval()
        seen_super_correct = 0
        seen_sub_correct = 0
        unseen_super_correct = 0
        unseen_sub_correct = 0
        seen_total = 0
        unseen_total = 0

        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                images, super_labels, sub_labels = data[0].to(self.device), data[1].to(self.device), data[3].to(self.device)

                super_outputs, sub_outputs = self.model(images)
                
                super_preds = detect_novel_class(super_outputs, energy_threshold_super, novel_index=3)
                sub_preds = detect_novel_class(sub_outputs, energy_threshold_sub, novel_index=87)

                seen_mask = super_labels != 3
                unseen_mask = super_labels == 3

                seen_super_correct += (super_preds[seen_mask] == super_labels[seen_mask]).sum().item()
                seen_sub_correct += (sub_preds[seen_mask] == sub_labels[seen_mask]).sum().item()
                seen_total += seen_mask.sum().item()

                unseen_super_correct += (super_preds[unseen_mask] == super_labels[unseen_mask]).sum().item()
                unseen_sub_correct += (sub_preds[unseen_mask] == sub_labels[unseen_mask]).sum().item()
                unseen_total += unseen_mask.sum().item()

        print(f"Seen Super Acc: {100 * seen_super_correct / seen_total:.2f}%")
        print(f"Seen Sub Acc: {100 * seen_sub_correct / seen_total:.2f}%")
        print(f"Unseen Super Acc: {100 * unseen_super_correct / max(1, unseen_total):.2f}%")
        print(f"Unseen Sub Acc: {100 * unseen_sub_correct / max(1, unseen_total):.2f}%")

    def test(self, save_to_csv=True, return_predictions=False, energy_threshold_super=1.5, energy_threshold_sub=2.0):
        # self.model.eval()
        if not self.test_loader:
            raise NotImplementedError('test_loader not specified')

        # Evaluate on test set, in this simple demo no special care is taken for novel/unseen classes
        test_predictions = {'image': [], 'superclass_index': [], 'subclass_index': []}

        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                images, img_names = data[0].to(self.device), data[1]
                super_outputs, sub_outputs = self.model(images)

                super_preds = detect_novel_class(super_outputs, energy_threshold_super, novel_index=3)
                sub_preds = detect_novel_class(sub_outputs, energy_threshold_sub, novel_index=87)

                test_predictions['image'].append(img_names[0])
                test_predictions['superclass_index'].append(super_preds.item())
                test_predictions['subclass_index'].append(sub_preds.item())

        test_predictions = pd.DataFrame(data=test_predictions)
        if save_to_csv:
            test_predictions.to_csv('test_predictions.csv', index=False)
        if return_predictions:
            return test_predictions


# In[112]:


class LogitAdjustmentLoss(torch.nn.Module):
    def __init__(self, prior):
        super().__init__()
        self.register_buffer('prior', torch.tensor(prior).log())

    def forward(self, logits, labels):
        adjusted_logits = logits + self.prior
        return F.cross_entropy(adjusted_logits, labels)


# In[113]:


# Estimate class frequencies from training set
super_freq = train_subset_df['superclass_index'].value_counts(normalize=True).sort_index().values
sub_freq = train_subset_df['subclass_index'].value_counts(normalize=True).sort_index().values

criterion_super = LogitAdjustmentLoss(super_freq)
criterion_sub = LogitAdjustmentLoss(sub_freq)


# In[114]:


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
criterion_super = LogitAdjustmentLoss(super_freq)
criterion_sub = LogitAdjustmentLoss(sub_freq)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
trainer = CLIPTrainer(model, criterion, optimizer, train_loader, val_loader, test_loader, device=device)


# In[115]:


# Training loop
# for epoch in range(1):
#     print(f'Epoch {epoch+1}')
#     trainer.train_epoch()
#     trainer.validate_epoch(energy_threshold_super=6.5, energy_threshold_sub=7.5)

#     print('')

# print('Finished Training')

# Train loop
for epoch in range(20):
    print(f"Epoch {epoch + 1}")
    trainer.train_epoch()
    trainer.validate_epoch(energy_threshold_super=6.5, energy_threshold_sub=7.5)

# Tune thresholds after training
print("Tuning energy thresholds...")
best_threshold_super, best_threshold_sub = tune_energy_threshold(model, val_loader, device=device)

# Final validation with best thresholds
print("Validation with best thresholds:")
trainer.validate_epoch(best_threshold_super, best_threshold_sub)

# Final test prediction with best thresholds
print("Test Prediction with best thresholds:")
test_predictions = trainer.test(save_to_csv=True,
                                return_predictions=True,
                                energy_threshold_super=best_threshold_super,
                                energy_threshold_sub=best_threshold_sub)


# In[ ]:


# test_predictions = trainer.test(save_to_csv=True, return_predictions=True)
# print("\nTest predictions saved to test_predictions.csv")


# In[ ]:


energy_scores = []
with torch.no_grad():
    for i, data in enumerate(val_loader):
        images, _, _, = data[0].to(device), data[1].to(device), data[3].to(device)
        super_outputs, sub_outputs = model(images)
        energy = torch.logsumexp(super_outputs, dim=1)
        energy_scores.extend(energy.cpu().numpy())

plt.hist(energy_scores, bins=50)
plt.title("Energy distribution")
plt.xlabel("Energy")
plt.ylabel("Frequency")
plt.show()


# In[ ]:





# In[ ]:




