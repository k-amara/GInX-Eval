from typing import List, Set, Dict, Optional, Union
import itertools
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

class Locator:

    def __init__(self, g_embeds, n_embeds, sgs, d_lr, isolated_nodes, device=None):

        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device

        self.isolated_nodes = isolated_nodes

        feat_dim = g_embeds.size()[1] + n_embeds.size()[-1]
        self.model = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        ).to(self.device)

        self.pretrain_loss_func = nn.KLDivLoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr= d_lr)
        self.sgs = sgs

        self.logit_embeds = []
        offset = 0
        for i, sg in enumerate(self.sgs):
            tn = n_embeds[offset:offset+len(sg),:]
            tg = g_embeds[i]
            tg = tg.repeat(tn.size(0), 1)
            self.logit_embeds.append(torch.cat((tg, tn),1))
            offset += len(sg)
        
        self.logit_embeds = torch.cat(self.logit_embeds).to(self.device).detach()
        


    def train(self, epochs, fn, score_fn, sample_rate=1):

        index = np.arange(len(self.sgs))
        # print('index',len(index))
        np.random.shuffle(index)
        selected_sg = [False]*len(self.sgs)
        n_selected_sg = int(sample_rate*len(self.sgs))
        for i in range(n_selected_sg):
            selected_sg[index[i]] = True
        
        self.model.train()
        all_rewards = []
        all_reward_label = []
        for i in range(len(self.sgs)):
            # print(i)
            if selected_sg[i]:
                nodes = list(self.sgs[i])
                
                nonisolated_nodes = []
                for n_id in nodes:
                    if n_id not in self.isolated_nodes:
                        nonisolated_nodes.append(n_id)
                part_generated_sgs = [x[:-1] if x[-1] == 'EOS' else x for x in fn(nonisolated_nodes,score_fn)]

                generated_sgs = []
                i = 0
                for n_id in nodes:
                    if n_id not in self.isolated_nodes:
                        generated_sgs.append(part_generated_sgs[i])
                        i+=1
                    else:
                        generated_sgs.append([n_id])

                rewards = score_fn(generated_sgs)
                rewards = torch.from_numpy(rewards).float().to(self.device)
                all_rewards.append(nn.Softmax(dim=0)(rewards))
                all_reward_label.append(torch.argmax(rewards).item())

        # print('test_here2')
        for i in range(epochs):
            logits = self.model(self.logit_embeds).squeeze()

            # all_logits = []
            offset = 0
            loss = 0
            r_i = 0
            for j in range(len(self.sgs)):
                if selected_sg[j]:
                    softmax_logits = nn.Softmax(dim=0)(logits[offset:offset+len(self.sgs[j])])
                    loss += self.pretrain_loss_func(softmax_logits, all_rewards[r_i])
                    r_i += 1

                offset += len(self.sgs[j])
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print("iter", i, "train_loss", loss.item()/r_i)
        
 
    def train_step(self, fn, score_fn):
        self.model.train()
        self.optimizer.zero_grad()
        all_logits = self.model(self.logit_embeds).squeeze()

        offset = 0
        seeds = []
        logps = []
        for sg in self.sgs:
            logits = torch.log_softmax(all_logits[offset:offset+len(sg)], 0)
            offset += len(sg)
            ps = torch.exp(logits.detach())
            seed_idx = torch.multinomial(ps, 1).item()
            seeds.append(sg[seed_idx])
            logps.append(logits[seed_idx])
        logps = torch.stack(logps)

        generated_sgs = [x[:-1] if x[-1] == 'EOS' else x for x in fn(seeds)]
        rewards = score_fn(generated_sgs)
        rewards = torch.from_numpy(rewards).float().to(self.device)

        policy_loss = -(rewards * logps).mean()
        policy_loss.backward()
        self.optimizer.step()

        return policy_loss.item(), rewards.mean().item()


    def get_eval_seed_stochastic(self):

        self.model.eval()
        with torch.no_grad():
            all_logits = self.model(self.logit_embeds).squeeze()

            offset = 0
            seeds = []
            for sg in self.sgs:
                logits = torch.log_softmax(all_logits[offset:offset+len(sg)], 0)
                offset += len(sg)
                ps = torch.exp(logits.detach())
                seed_idx = torch.multinomial(ps, 1).item()
                seeds.append(sg[seed_idx])
        
        return seeds


    def get_eval_seed_deterministic(self):

        self.model.eval()
        with torch.no_grad():
            all_logits = self.model(self.logit_embeds).squeeze()

            offset = 0
            seeds = []
            for sg in self.sgs:
                logits = torch.log_softmax(all_logits[offset:offset+len(sg)], 0)
                offset += len(sg)

                seed_id_sort = torch.argsort(logits.detach(), descending=True)
                flag = False
                for seed_idx in seed_id_sort:
                    if sg[seed_idx] not in self.isolated_nodes:
                        seeds.append(sg[seed_idx])
                        flag = True
                        break
                if not flag:
                    print("There exists an isolated graph!!!")
                
    
        return seeds
