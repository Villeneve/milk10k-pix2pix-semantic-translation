import torch
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import v2 as tt

import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

class Data(Dataset):
    def __init__(self,transforms=None):
        super().__init__()
        self.transforms = transforms
        self.imagePath = '/storage/SSD1/.data/milk10k/images/'
        df = pd.read_csv('/storage/SSD1/.data/milk10k/supplements/training_input.csv')
        df = df[['lesion_id','image_type','isic_id']]
        df_derm = df[df['image_type']=='dermoscopic']
        df_clin = df[df['image_type']=='clinical: close-up']
        self.df = pd.merge(df_derm,df_clin,on='lesion_id')

    def __getitem__(self, index):
        derm = self.imagePath + self.df.iloc[index]['isic_id_x'] + '.jpg'
        clin = self.imagePath + self.df.iloc[index]['isic_id_y'] + '.jpg'
        derm = Image.open(derm)
        clin = Image.open(clin)
        if self.transforms is not None:
            return self.transforms(derm),self.transforms(clin)
        return derm,clin

    def __len__(self):
        return len(self.df)

def load_dataset(batch_size: int, shuffle: bool, num_workers: int):
    dataset_dict = torch.load('data/dataset.data')
    dataset = torch.utils.data.TensorDataset(
        dataset_dict['derm'],
        dataset_dict['clin']
    )
    dataset = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dataset

def setXavier_(layer):
    torch.nn.init.xavier_normal_(layer.weight)
    torch.nn.init.zeros_(layer.bias)
    return layer

def setKaiming_(layer,alpha=0.0):
    torch.nn.init.kaiming_normal_(layer.weight,alpha)
    torch.nn.init.zeros_(layer.bias)
    return layer

def plot_sample(model:torch.nn.Module,dataset:torch.utils.data.DataLoader, transform, N=7):
    device = next(model.parameters()).device
    derm,_ = next(iter(dataset))
    derm = transform(derm).to(device)
    with torch.inference_mode():
        output = model(derm).permute(0,2,3,1).cpu().numpy()/2+1/2
    fig,ax = plt.subplots(N,N,figsize=(2000/300,2000/300),dpi=300)
    ax = ax.ravel()
    for i in range(N**2):
        ax[i].imshow(output[i])
        ax[i].axis(False)
    plt.tight_layout(pad=0)
    plt.savefig('sample.png')
    plt.close()
    return

def setGrad_(model:torch.nn.Module, state:bool) -> None:
    for p in model.parameters():
        p.requires_grad_(state)
    return

if __name__ == '__main__':
    data = Data()
    # data = DataLoader(data,1,False,num_workers=4)
    dataset_dict = {
        'derm':[],
        'clin':[]
    }
    compose = tt.Compose([
        tt.Resize(256,interpolation=tt.InterpolationMode.BOX),
        tt.ToImage(),
        tt.ToDtype(torch.uint8),
        tt.CenterCrop((256,256)),
    ])
    for derm,clin in tqdm(data):
        derm = compose(derm)
        clin = compose(clin)
        dataset_dict['derm'].append(derm)
        dataset_dict['clin'].append(clin)
    dataset_dict['derm'] = torch.stack(dataset_dict['derm'],0)
    dataset_dict['clin'] = torch.stack(dataset_dict['clin'],0)
    os.makedirs('data',exist_ok=True)
    torch.save(dataset_dict,'data/dataset.data')