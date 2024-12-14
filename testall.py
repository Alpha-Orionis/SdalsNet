import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO
import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
from torchvision import datasets
import numpy as np
from PIL import Image
import utils
import vision_transformer as vits
from torch.utils.data.dataloader import DataLoader
from PIL import Image
from Eval import evaluate

def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return


def test(args):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model
    model = vits.__dict__['vit_small'](patch_size=8, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    transform = pth_transforms.Compose([
        pth_transforms.Resize((480, 480)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    #state_dict = torch.load('/data1/shoupeiyao/workspace/UCOD/last/checkpoint' + args.epoch + '.pth', map_location="cpu")
    state_dict = torch.load('/data1/shoupeiyao/workspace/UCOD/models/0.6-0.6-0.4.pth', map_location="cpu")
    if "teacher" in state_dict:
        state_dict = state_dict["teacher"]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)

    lis = ['CHAMELEON','CAMO','COD10K','NC4K']

    for name in lis:
        path = '/data1/shoupeiyao/Data/dirnetdata/datas/TestDataset/' + name + '/Imgs/'
        namelist = os.listdir(path)
        print('************', name, '**************')
        for i in range(len(namelist)):
            img = Image.open(path + namelist[i])
            img = img.convert('RGB')
            W_, H_ = img.size
            img = transform(img)
            w, h = img.shape[1] - img.shape[1] % 8, img.shape[2] - img.shape[2] % 8
            img = img[:, :w, :h].unsqueeze(0)
    
            w_featmap = img.shape[-2] // 8
            h_featmap = img.shape[-1] // 8
            x, attentions, out = model(img.to(device))
    
            nh = attentions.shape[1]  # number of head
            attn = attentions[0, :, 0, 1:]
            _, L = attn.shape
            H = W = int(pow(L, 0.5))

            attn = attn[0]
            attn = torch.where(attn < attn.mean(dim=-1, keepdim=True).expand(1, L), 0., 1.)
            attn = attn.view(1, 1, H, W)
            
            last = attn
            last2 = nn.functional.avg_pool2d(last, 33, 1, 16)
            last =  nn.functional.avg_pool2d(last, 5, 1, 2)
            last =  torch.where(last < 0.5, 0., 1.)
            last2 = torch.where(last2 < last2.mean(), 0., 1.)
            last *= last2

            last = nn.functional.interpolate(last.cpu(), size=(H_, W_), mode="nearest").squeeze().squeeze().int().numpy()
    
            Image.fromarray((last * 255).astype(np.uint8)).save(os.path.join('/data1/shoupeiyao/workspace/UCOD/last/outs/' + name + '/', namelist[i][:-4] + '.png'))
        
        evaluate(name)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--epoch', default='0012', type=str, help='Patch resolution of the model.')
    args = parser.parse_args()
    test(args)