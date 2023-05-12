import os
import sys

import numpy as np
import scipy
import torch
from tqdm import tqdm
from collections import defaultdict

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from psr_task import data_loader, pl_module

ckpt_path = "epoch=99-step=2540000.ckpt"
# checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
# model_weights = checkpoint["state_dict"]
# model = models.PSRSurfNet()
# model.load_state_dict(model_weights)

model = pl_module.PSRModule.load_from_checkpoint(ckpt_path, map_location='cpu')
# model = pl_module.PSRModule.load_from_checkpoint(ckpt_path)
model.eval()

data_dir = '../data/PSR/test/'
dataset = data_loader.PSRDataset(data_dir)
loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=0, collate_fn=lambda x: x)
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

all_results = defaultdict(list)
all_lists = []
for i, item in tqdm(enumerate(loader), total=len(loader)):
    # if i > 2:
    #     break
    with torch.no_grad():
        name, geom_feats, scores = item[0]
        if name is None:
            print('None here')
            continue
        output = model(geom_feats)
        scores = scores.item()
        output = output.item()
        reslist = [output, scores]
        all_lists.append(reslist)
        all_results[name[:4]].append(reslist)

all_lists = np.array(all_lists)
global_r = scipy.stats.spearmanr(all_lists, axis=0).statistic
local_r = []
for system, lists in all_results.items():
    lists = np.array(lists)
    r = scipy.stats.spearmanr(lists, axis=0).statistic
    local_r.append(r)
local_r = np.mean(local_r)

print(f" Global R : {global_r}")
print(f" Local R : {local_r}")
