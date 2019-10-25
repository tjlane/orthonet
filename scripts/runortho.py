#!/usr/bin/env python

from __future__ import print_function

import os
import h5py
import argparse
import numpy as np

import torch
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.utils import save_image

from orthonet.orthoae import OrthoEnc
from orthonet.utils import binary_normalize
from orthonet.jacob import JG_MSE_Loss
from orthonet import visual

# ---------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='run VAE on simple simulation')
parser.add_argument('data_file', type=str, help='data file to load')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

# ---------------------------------------------------------------------------------

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device_type = "cuda" if args.cuda else "cpu"
print("Device: %s" % device_type)
device = torch.device(device_type)
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

if not os.path.exists('results'):
    os.mkdir('results')

# ---------------------------------------------------------------------------------

f = h5py.File(args.data_file)
data = binary_normalize( np.array(f['data']).astype(np.float32) )
f.close()

data = data.reshape(-1, 33*33)

split = int(data.shape[0] * 0.9)
size  = data.shape[0]
field_shape = data.shape[1:]

print('\tTrain/Test: %d/%d' % (split, size-split))

train_data = torch.stack([torch.tensor(data[i]) for i in range(0, split)])
test_data  = torch.stack([torch.tensor(data[i]) for i in range(split, size)])

train_loader = torch.utils.data.DataLoader( 
        torch.utils.data.TensorDataset(train_data),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_data),
    batch_size=args.batch_size, shuffle=True, **kwargs)

# ---------------------------------------------------------------------------------

model = OrthoEnc().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.1)
loss_fxn = JG_MSE_Loss(beta=10.0)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data,) in enumerate(train_loader):

        data = data.to(device)
        recon_batch = model(data)

        loss = loss_fxn(recon_batch, data, data, model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

    return train_loss  / len(train_loader.dataset)


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data,) in enumerate(test_loader):

            data = data.to(device)
            recon_batch = model(data)

            test_loss += loss_fxn(recon_batch, data, data, model).item()
            if i == 0:
                n = min(data.size(0), 8)
                com_shape = (args.batch_size, 1, field_shape[0], field_shape[1])
                comparison = torch.cat([data.view(*com_shape)[:n],
                                      recon_batch.view(*com_shape)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

    return test_loss


if __name__ == "__main__":

    test_loss  = np.zeros(args.epochs)
    train_loss = np.zeros(args.epochs)

    for epoch in range(1, args.epochs + 1):

        test_loss[epoch-1]  = train(epoch)
        train_loss[epoch-1] = test(epoch)

        with torch.no_grad():
            sample = torch.randn(64, 4).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, field_shape[0], field_shape[1]),
                       'results/sample_' + str(epoch) + '.png')

    visual.plot_loss_curves(test_loss, train_loss, save='results/loss_curves.png')


