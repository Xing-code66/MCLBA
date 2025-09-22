import sys, os
from random import random
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch_geometric.graphgym import optim
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.abspath('..'))
import torch.optim.lr_scheduler as lr_scheduler
from utils.datareader import GraphData, DataReader
from utils.batch import collate_batch
from model.gcn import GCN
from model.gat import GAT
from model.gin import GIN
from config import parse_args
args = parse_args()
seed = args.seed

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_clean_model(args, gdata_train, gdata_test, in_dim, out_dim):
    # assert torch.cuda.is_available(), 'no GPU available'
    # cpu = torch.device('cpu')
    # cuda = torch.device('cuda')

    loaders = {}
    loader_train = DataLoader(gdata_train, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    loader_test = DataLoader(gdata_test, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    loaders['train'] = loader_train
    loaders['test'] = loader_test

    if args.model == 'gcn':
        model = GCN(in_dim, out_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
    elif args.model == 'gat':
        model = GAT(in_dim, out_dim, hidden_dim=args.hidden_dim, dropout=args.dropout, num_head=args.num_head)
    elif args.model == 'gin':
        model = GIN(in_dim, out_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
    else:
        raise NotImplementedError(args.model)

    # train_params = list(filter(lambda p: p.requires_grad, model.parameters()))

    loss_fn = F.cross_entropy
    predict_fn = lambda output: output.max(1, keepdim=True)[1].detach().cpu()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.5, 0.999))
    scheduler = lr_scheduler.MultiStepLR(optimizer, args.lr_decay_steps, gamma=0.1)

    model.to(device)
    for epoch in tqdm(range(args.train_clean_epochs)):
        model.train()
        # train_loss, n_samples = 0, 0
        for batch_id, data in enumerate(loaders['train']):
            for i in range(len(data)):
                data[i] = data[i].to(device)

            optimizer.zero_grad()
            output = model(data)

            if len(output.shape) == 1:
                output = output.unsqueeze(0)

            loss = loss_fn(output, data[4])
            loss.to(device)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # train_loss += loss.item() * len(output)
            # n_samples += len(output)

        if (epoch + 1) % args.eval_every == 0 or epoch == args.train_epochs - 1:
            model.eval()

            test_loss, correct, n_samples = 0, 0, 0
            for batch_id, data in enumerate(loaders['test']):
                for i in range(len(data)):
                    data[i] = data[i].to(device)
                # if args.use_org_node_attr:
                #     data[0] = norm_features(data[0])
                output = model(data)
                if len(output.shape) == 1:
                    output = output.unsqueeze(0)
                loss = loss_fn(output, data[4], reduction='sum')
                test_loss += loss.item()
                n_samples += len(output)
                pred = predict_fn(output)

                correct += pred.eq(data[4].detach().cpu().view_as(pred)).sum().item()
            eval_acc = 100. * correct / n_samples
            # print('Clean model test clean samples：Test set (epoch %d): Average loss: %.4f, Accuracy: %d/%d (%.2f%s)' % (epoch + 1, test_loss / n_samples, correct, n_samples, eval_acc, '%'))

    model.to(device)

    return model, eval_acc

