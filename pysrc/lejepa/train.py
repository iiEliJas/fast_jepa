import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

from .model import LeJEPAEncoder, Projector
from .sigreg import SlicedEppsPulley
from .loss import lejepa_loss
from .diagnostics import embedding_diagnostics


def build_transforms(img_size=128):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])


def train_one_epoch(model, projector, sigreg, optimizer, loader, device, lamb=0.02, V=4):
    model.train(); projector.train()
    for images, _ in loader:
        images = images.to(device)
        # create V views by repeating simple transform not per-view; placeholder
        views = images.repeat(V, 1, 1, 1)
        emb = model(views)
        z = projector(emb).view(V, images.size(0), -1)
        loss, lp, ls = lejepa_loss(z, sigreg, lamb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()