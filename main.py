#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        (.5,.99),
    ),
    torch.optim.Adam(
        crit.parameters(),
        2e-4,
        (.5,.99)
    )
]
rec_loss_g = None
adv_loss_g = None
adv_loss_c = None
cut_loss = None

#%%
for epoch in range(5000):
    os.makedirs('weights',exist_ok=True)
    torch.save(gen.state_dict(),'weights/gen.weights')
    if epoch%1==0: plot_sample(gen,dataset,tr,N=5)
    batch_graph = tqdm(dataset)
    for i,(derm,clin) in enumerate(batch_graph):
        derm = tr(derm).to(gpu)
        clin = tr(clin).to(gpu)

        setGrad_(crit,True)
        with torch.no_grad():
            fake_clin = gen(derm)
        true_logits = crit(torch.cat([derm,clin],1))
        fake_logits = crit(torch.cat([derm,fake_clin],1))
        adv_loss_c_ = F.relu(-true_logits+1).mean() + F.relu(fake_logits+1).mean()
        if i%16==0:
            derm_grad = torch.cat([derm,clin],1).clone().requires_grad_(True)
            grad = torch.autograd.grad(
                crit(derm_grad).flatten(1).mean(1).sum(),
                derm_grad,
                create_graph=True,
                only_inputs=True,
            )[0]
            norm = grad.flatten(1).square().sum(1)
            adv_loss_c_ += 16*5*norm.mean()
        opt[1].zero_grad()
        adv_loss_c_.backward()
        opt[1].step()


        setGrad_(crit,False)
        fake_clin = gen(derm)
        cut_fake = gen.encoder(fake_clin).flatten(2).permute(0,2,1)
        cut_fake = F.normalize(cut_fake,2,-1)
        with torch.no_grad():
            cut_true = F.normalize(gen.encoder(derm).flatten(2),2,1)
        matrix = (cut_fake@cut_true)/0.1
        N, HW, _ = matrix.size()
        matrix = matrix.view(N*HW,HW)
        cut_loss_ = F.cross_entropy(matrix,torch.arange(256,device=gpu).repeat(N))

        idt_clin = gen(clin)
        cut_fake_idt = F.normalize(gen.encoder(idt_clin).flatten(2).permute(0,2,1), 2, -1)
        with torch.no_grad():
            cut_true_idt = F.normalize(gen.encoder(clin).flatten(2), 2, 1)
        matrix_idt = (cut_fake_idt @ cut_true_idt) / 0.1
        N2, HW2, _ = matrix_idt.shape
        target_idt = torch.arange(HW2, device=gpu).repeat(N2)
        cut_loss_Y = F.cross_entropy(matrix_idt.reshape(N2*HW2, HW2), target_idt)
        cut_loss_ += cut_loss_Y

        fake_logits = crit(torch.cat([derm,fake_clin],1))
        # rec_loss_g_ = 100*(clin-fake_clin).abs().mean()
        adv_loss_g_ = -fake_logits.mean()
        opt[0].zero_grad()
        (adv_loss_g_+cut_loss_).backward()
        opt[0].step()


        # rec_loss_g = rec_loss_g_.item() if rec_loss_g is None else .98*rec_loss_g+(1-.98)*rec_loss_g_.item()
        adv_loss_g = adv_loss_g_.item() if adv_loss_g is None else .98*adv_loss_g+(1-.98)*adv_loss_g_.item()
        adv_loss_c = adv_loss_c_.item() if adv_loss_c is None else .98*adv_loss_c+(1-.98)*adv_loss_c_.item()
        cut_loss = cut_loss_.item() if cut_loss is None else .98*cut_loss+(1-.98)*cut_loss_.item()
        batch_graph.set_postfix({
            # 'rec_loss_g':f'{rec_loss_g:.4f}',
            'adv_loss_g':f'{adv_loss_g:.4f}',
            'adv_loss_c':f'{adv_loss_c:.4f}',
            'cut_loss':f'{cut_loss:.4f}'
        })
        batch_graph.set_description(f'Ep: {epoch}')
        # if i==9: break

#%%
plot_sample(gen,dataset,tr,N=5)