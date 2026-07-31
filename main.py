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
dataset = load_dataset(64,False,4)

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
rec_loss_g = None
adv_loss_g = None
adv_loss_c = None

#%%
for epoch in range(5000):
    os.makedirs('weights',exist_ok=True)
    torch.save(gen.state_dict(),'weights/gen.weights')
    if epoch%1==0: plot_sample(gen,dataset,tr,N=5)
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
            rec_loss_g_ = (clin-fake_clin).abs().mean()
            adv_loss_g_ = -fake_logits.mean()
            opt[0].zero_grad()
            (rec_loss_g_+adv_loss_g_).backward()
            opt[0].step()


        rec_loss_g = rec_loss_g_.item() if rec_loss_g is None else .98*rec_loss_g+(1-.98)*rec_loss_g_.item()
        adv_loss_g = adv_loss_g_.item() if adv_loss_g is None else .98*adv_loss_g+(1-.98)*adv_loss_g_.item()
        adv_loss_c = adv_loss_c_.item() if adv_loss_c is None else .98*adv_loss_c+(1-.98)*adv_loss_c_.item()
        batch_graph.set_postfix({
            'rec_loss_g':f'{rec_loss_g:.4f}',
            'adv_loss_g':f'{adv_loss_g:.4f}',
            'adv_loss_c':f'{adv_loss_c:.4f}'
        })
        batch_graph.set_description(f'Ep: {epoch}')
        # if i==9: break

#%%
plot_sample(gen,dataset,tr,N=5)