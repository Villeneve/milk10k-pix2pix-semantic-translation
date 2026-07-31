#%%
import torch
import torch.nn as nn

from torchinfo import summary

from src.utils import *
from src.layers import *
from src.models import Critic, Generator

import matplotlib.pyplot as plt
import numpy as np
from tqdm.autonotebook import tqdm

#%%
tr = tt.Compose([
    tt.ToDtype(torch.float32,scale=True),
    tt.Normalize([.5]*3,[.5]*3)
])
dataset = load_dataset(32,True,4)

#%%
gpu = torch.device('cuda:1')
crit = Critic(depth=5,max_ch=512).to(gpu)
gen = Generator(8).to(gpu)

opt = [
    torch.optim.Adam(
        gen.parameters(),
        2e-4,
        (0.,.9),
    ),
    torch.optim.Adam(
        crit.parameters(),
        2e-4,
        (0.,.9)
    )
]

#%%
for epoch in range(5000):
    os.makedirs('weights',exist_ok=True)
    torch.save(gen.state_dict(),'weights/gen.weights')
    plot_sample(gen,dataset,tr,N=5)
    batch_graph = tqdm(dataset)
    for i,(derm,clin) in enumerate(batch_graph):
        derm = tr(derm).to(gpu)
        clin = tr(clin).to(gpu)

        with torch.no_grad():
            fake_clin = gen(derm)
        true_logits = crit(torch.cat([derm,clin],1))
        fake_logits = crit(torch.cat([derm,fake_clin],1))
        eps_drift = 1e-3*(true_logits**2).mean()
        adv_loss_c_ = -true_logits.mean() + fake_logits.mean() + eps_drift
        opt[1].zero_grad()
        adv_loss_c_.backward()
        opt[1].step()

        if i%5==0:
            fake_clin = gen(derm)
            fake_logits = crit(torch.cat([derm,fake_clin],1))
            rec_loss_g_ = 100*(clin-fake_clin).abs().mean()
            adv_loss_g_ = -fake_logits.mean()
            opt[0].zero_grad()
            (rec_loss_g_+adv_loss_g_).backward()
            opt[0].step()

        batch_graph.set_postfix({
            'rec_loss_g_':f'{rec_loss_g_.item():.4f}',
            'adv_loss_g_':f'{adv_loss_g_.item():.4f}',
            'adv_loss_c_':f'{adv_loss_c_.item():.4f}'
        })

#%%
plot_sample(gen,dataset,tr,N=5)