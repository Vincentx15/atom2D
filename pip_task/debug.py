import time
import os
import sys

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from reprocess_data import NewPIP
from torch.utils.data import DataLoader

train_data_dir = "../data/PIP/DIPS-split/data/train"
val_data_dir = "../data/PIP/DIPS-split/data/val"
train_dataset = NewPIP(data_dir=train_data_dir,
                       geometry_path='../data/processed_data/geometry/',
                       operator_path='../data/processed_data/operator/')
val_dataset = NewPIP(data_dir=val_data_dir,
                     geometry_path='../data/processed_data/geometry/',
                     operator_path='../data/processed_data/operator/')

train_loader = DataLoader(train_dataset, num_workers=6, batch_size=1, collate_fn=lambda x: x)
val_loader = DataLoader(val_dataset, num_workers=6, batch_size=1, collate_fn=lambda x: x)

t0 = time.time()
for i, batch in enumerate(train_loader):
    if i % 1000:
        print(f"Train : done {i}/{len(train_loader)} steps in {time.time() - t0}")

t0 = time.time()
for i, batch in enumerate(train_loader):
    if i % 1000:
        print(f"Validation : done {i}/{len(train_loader)} steps in {time.time() - t0}")
