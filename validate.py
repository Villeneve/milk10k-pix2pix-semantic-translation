#%%
import torch

from torchvision.transforms import v2 as tt
from src.utils import *
from src.layers import *
from src.models import Generator

import matplotlib.pyplot as plt

#%%
gpu = torch.device('cuda:0')
dataset = load_dataset(32,True,4)
tr = tt.Compose([
    tt.ToDtype(torch.float32,scale=True),
    tt.Normalize([.5]*3,[.5]*3)
])

gen = Generator(8).to(gpu)
gen.load_state_dict(torch.load('weights/gen.weights'))

#%%
for derm,clin in dataset:
    derm = tr(derm).to(gpu)
    clin = tr(clin)
    with torch.inference_mode():
        output = gen(derm)[0].permute(1,2,0).cpu().numpy()/2+1/2
    plt.subplot(1,3,1)
    plt.imshow(derm[0].permute(1,2,0).cpu().numpy()/2+1/2)
    plt.title('input')
    plt.axis(False)
    plt.subplot(1,3,2)
    plt.imshow(clin[0].permute(1,2,0).numpy()/2+1/2)
    plt.title('ground')
    plt.axis(False)
    plt.subplot(1,3,3)
    plt.imshow(output)
    plt.title('output')
    plt.axis(False)
    plt.tight_layout()
    plt.show()
    break