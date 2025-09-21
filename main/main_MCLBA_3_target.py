import random
import sys,  os
from matplotlib import pyplot as plt
from utils.defense_method import Prune_low_similarity, Add_noise

sys.path.append(os.path.abspath('..'))
import time
from utils.functions import *
from utils.train_clean_model import train_clean_model
from utils.datareader import GraphData, DataReader
from utils.batch import collate_batch
from model.gcn import GCN
from model.gat import GAT
from model.gin import GIN
import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from utils.datareader import DataReader
from config import parse_args
from torch.utils.tensorboard import SummaryWriter


def run(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    for data_i in ['ENZYMES', 'IMDB-MULTI']:
        args.dataset = data_i
        print('\n')
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++dataset:{}++++++++++++++++++++++++++++++++++++++++++++++++++++++++'.format(data_i))
        dr = DataReader(args)

        nodenums = [adj.shape[0] for adj in dr.data['adj_list']]
        nodemax = max(nodenums)
        featdim = np.array(dr.data['features'][0]).shape[1]
        print('featdim:', featdim)
        num_class = dr.data["num_classes"]

        for split in ['train', 'test']:
            if split == 'train':
                gids = dr.data['splits']['train']
                gdata_train = GraphData(dr, gids)
            else:
                gids = dr.data['splits']['test']
                gdata_test = GraphData(dr, gids)

        clean_gdata_train_1 = copy.deepcopy(gdata_train)
        clean_gdata_test_1 = copy.deepcopy(gdata_test)

        in_dim = featdim
        out_dim = num_class

        if args.backdoor == True:
            print("The type of selecting sample：{}-----------------".format(args.select_poison_graph))
            if args.select_poison_graph == "target_class_random":
                poison_graph_idx_0, poison_graph_idx_1, poison_graph_idx_2 = Select_poison_graph_idx_3_target(args, gdata_train)
            print('datasets:{}.  num of poisoned graphs:{}. \n is ：{}'.format(args.dataset, len(poison_graph_idx_0+poison_graph_idx_1+poison_graph_idx_2), poison_graph_idx_0+poison_graph_idx_1+poison_graph_idx_2 ))

            # select poisoned node in poisoned graph
            print('The method of selecting nodes：{}-----------------'.format(args.select_poison_node))
            print('The num of poison node: {}'.format(args.bkd_size))
            if args.select_poison_node == 'random':
                nidx_train_dict_0 = Rand_select_node_idx(args, gdata_train, poison_graph_idx_0)
                nidx_train_dict_1 = Rand_select_node_idx(args, gdata_train, poison_graph_idx_1)
                nidx_train_dict_2 = Rand_select_node_idx(args, gdata_train, poison_graph_idx_2)

            np.random.seed(args.seed)
            feature_trigger_0 = np.random.uniform(-2, 0, args.trig_size)
            print('trigger 0 :', feature_trigger_0)
            feature_trigger_1 = np.random.uniform(-5,  -3, args.trig_size)
            print('trigger 1 :', feature_trigger_1)
            feature_trigger_2 = np.random.uniform(1, 3, args.trig_size)
            print('trigger 2 :', feature_trigger_2)
            print('trigger size：{}'.format(args.trig_size))

            trigger_0_inj_position, trigger_1_inj_position, trigger_2_inj_position = Select_trigger_inj_position_3_target_class(args, gdata_train)

            # -------------injecting trigger to training set， Do not modify label ---------------
            gdata_train = inject_train_trigger(args, gdata_train, poison_graph_idx_0, nidx_train_dict_0,
                                                   feature_trigger_0, trigger_0_inj_position)
            gdata_train = inject_train_trigger(args, gdata_train, poison_graph_idx_1, nidx_train_dict_1,
                                                   feature_trigger_1, trigger_1_inj_position)
            gdata_train = inject_train_trigger(args, gdata_train, poison_graph_idx_2, nidx_train_dict_2,
                                                   feature_trigger_2, trigger_2_inj_position)

            if args.if_use_defense == True:
                if args.defense_method == "low_sim_prune":
                    print("Prune low sim edge in training set......")
                    gdata_train = Prune_low_similarity(args, gdata_train)
                    print("Prune completed---------")
                elif args.defense_method == 'noise':
                    print("Adding noise to poisoned training set")
                    gdata_train = Add_noise(args, gdata_train)

            # -----------------------------inject trigger into all test sets------------------------------
            gdata_test_0 = copy.deepcopy(gdata_test)
            gdata_test_1 = copy.deepcopy(gdata_test)
            gdata_test_2 = copy.deepcopy(gdata_test)

            poison_test_idx = list(range(len(gdata_test.adj_list)))
            nidx_test_dict = Rand_select_node_idx(args, gdata_test, poison_test_idx)


            for test_i in [0, 1, 2]:
                args.test_trig_class = test_i
                if args.test_trig_class == 0:
                    poi_class = 0
                    gdata_test_0 = inject_test_trigger(gdata_test_0, poison_test_idx, nidx_test_dict, feature_trigger_0,
                                                       poi_class, trigger_0_inj_position)

                elif args.test_trig_class == 1:
                    poi_class = 1
                    gdata_test_1 = inject_test_trigger(gdata_test_1, poison_test_idx, nidx_test_dict, feature_trigger_1,
                                                       poi_class, trigger_1_inj_position)

                elif args.test_trig_class == 2:
                    poi_class = 2
                    gdata_test_2 = inject_test_trigger(gdata_test_2, poison_test_idx, nidx_test_dict, feature_trigger_2,
                                                       poi_class, trigger_2_inj_position)

        loader_train = DataLoader(gdata_train, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
        loader_test = DataLoader(gdata_test, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

        loader_test_0 = DataLoader(gdata_test_0, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
        loader_test_1 = DataLoader(gdata_test_1, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
        loader_test_2 = DataLoader(gdata_test_2, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

        print('train %d, test %d' % (len(loader_train.dataset), len(loader_test.dataset)))

        for model_i in ['gcn', 'gat', 'gin']:  # 'gcn', 'sage', 'gat', 'gin'
            args.model = model_i
            # prepare model
            if args.model == 'gcn':
                model = GCN(in_dim, out_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
            elif args.model == 'gat':
                model = GAT(in_dim, out_dim, hidden_dim=args.hidden_dim, dropout=args.dropout, num_head=args.num_head)
            elif args.model == 'gin':
                model = GIN(in_dim, out_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
            else:
                raise NotImplementedError(args.model)

            print('\n')
            print('----------------------model:{}----------------------'.format(args.model))

            # ------------train clean model--------------
            # clean_model = copy.deepcopy(model)
            print('train clean model --{}--.'.format(model_i))
            clean_model, clean_acc = train_clean_model(args, clean_gdata_train_1, clean_gdata_test_1, in_dim, out_dim)
            print('training clean model --{}-- completed！ Acc：{}'.format(model_i, clean_acc))

            loss_fn = F.cross_entropy
            predict_fn = lambda output: output.max(1, keepdim=True)[1].detach().cpu()
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.5, 0.999))
            scheduler = lr_scheduler.MultiStepLR(optimizer, args.lr_decay_steps, gamma=0.1)
            model.to(device)

            writer = SummaryWriter('logs')
            for epoch in tqdm(range(args.train_epochs)):
                model.train()
                start = time.time()
                train_loss, n_samples = 0, 0
                for batch_id, data in enumerate(loader_train):
                    for i in range(len(data)):
                        data[i] = data[i].to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    if len(output.shape) == 1:
                        output = output.unsqueeze(0)
                    loss = loss_fn(output, data[4])
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    time_iter = time.time() - start
                    train_loss += loss.item() * len(output)
                    n_samples += len(output)
                writer.add_scalar('{}_{}_training_loss'.format(args.dataset, args.model), train_loss/n_samples, epoch)  # 平均损失
                if args.train_verbose and (epoch % args.log_every == 0 or epoch == args.train_epochs - 1):
                    print('Train Epoch: %d\tLoss: %.4f (avg: %.4f) \tsec/iter: %.2f' % (
                        epoch + 1, loss.item(), train_loss / n_samples, time_iter / (batch_id + 1)))

                model.eval()
                test_loss_0, n_samples_0, non_target_samples_0, correct_non_0, total_pred_target_0 = 0, 0, 0, 0, 0
                for batch_id, data in enumerate(loader_test_0):
                    for i in range(len(data)):
                        data[i] = data[i].to(device)
                    output = model(data)
                    if len(output.shape) == 1:
                        output = output.unsqueeze(0)
                    loss_0 = loss_fn(output, data[4])
                    test_loss_0 += loss_0.item()
                    n_samples_0 += len(output)
                    pred = predict_fn(output)

                    labels = data[4].detach().cpu()
                    non_target_mask_0 = labels != args.target_class_0
                    non_target_samples_0 += non_target_mask_0.sum().item()

                    correct_non_target_0 = (pred[non_target_mask_0] == args.target_class_0).sum().item()
                    correct_non_0 += correct_non_target_0

                    pred_target_count_0 = (pred == args.target_class_0).sum().item()
                    total_pred_target_0 += pred_target_count_0

                if non_target_samples_0 > 0:
                    non_target_attack_acc_0 = 100. * correct_non_0 / non_target_samples_0
                else:
                    non_target_attack_acc_0 = 0
                overall_pred_target_acc_0 = 100. * total_pred_target_0 / n_samples_0

                # ++++++++++++++++++ Target  Class 1++++++++++++++++++++++++++
                test_loss_1, n_samples_1, non_target_samples_1, correct_non_1, total_pred_target_1 = 0, 0, 0, 0, 0
                for batch_id, data in enumerate(loader_test_1):
                    for i in range(len(data)):
                        data[i] = data[i].to(device)
                    output = model(data)
                    if len(output.shape) == 1:
                        output = output.unsqueeze(0)
                    loss_1 = loss_fn(output, data[4])
                    test_loss_1 += loss_1.item()
                    n_samples_1 += len(output)
                    pred = predict_fn(output)

                    labels = data[4].detach().cpu()
                    non_target_mask_1 = labels != args.target_class_1
                    non_target_samples_1 += non_target_mask_1.sum().item()

                    correct_non_target_1 = (pred[non_target_mask_1] == args.target_class_1).sum().item()
                    correct_non_1 += correct_non_target_1

                    pred_target_count_1 = (pred == args.target_class_1).sum().item()
                    total_pred_target_1 += pred_target_count_1

                if non_target_samples_1 > 0:
                    non_target_attack_acc_1 = 100. * correct_non_1 / non_target_samples_1
                else:
                    non_target_attack_acc_1 = 0

                overall_pred_target_acc_1 = 100. * total_pred_target_1 / n_samples_1


                # ++++++++++++++++++ Target  Class 1+++++++++++++++++++++++++++
                test_loss_2, n_samples_2, non_target_samples_2, correct_non_2, total_pred_target_2 = 0, 0, 0, 0, 0
                for batch_id, data in enumerate(loader_test_2):
                    for i in range(len(data)):
                        data[i] = data[i].to(device)
                    output = model(data)
                    if len(output.shape) == 1:
                        output = output.unsqueeze(0)
                    loss_2 = loss_fn(output, data[4])
                    test_loss_2 += loss_2.item()
                    n_samples_2 += len(output)
                    pred = predict_fn(output)

                    labels = data[4].detach().cpu()
                    non_target_mask_2 = labels != args.target_class_2
                    non_target_samples_2 += non_target_mask_2.sum().item()

                    correct_non_target_2 = (pred[non_target_mask_2] == args.target_class_2).sum().item()
                    correct_non_2 += correct_non_target_2

                    pred_target_count_2 = (pred == args.target_class_2).sum().item()
                    total_pred_target_2 += pred_target_count_2

                if non_target_samples_2 > 0:
                    non_target_attack_acc_2 = 100. * correct_non_2 / non_target_samples_2
                else:
                    non_target_attack_acc_2 = 0

                overall_pred_target_acc_2 = 100. * total_pred_target_2 / n_samples_2


            writer.close()

            # ---------The accuracy of backdoor model on clean sample--------------
            n_samples_2, correct_2 = 0, 0
            loader_clean_test = DataLoader(clean_gdata_test_1, batch_size=args.batch_size, shuffle=False,
                                           collate_fn=collate_batch)
            model.eval()
            for batch_id, data in enumerate(loader_clean_test):
                for i in range(len(data)):
                    data[i] = data[i].to(device)
                output = model(data)
                if len(output.shape) == 1:
                    output = output.unsqueeze(0)
                n_samples_2 += len(output)
                pred = predict_fn(output)
                correct_2 += pred.eq(data[4].detach().cpu().view_as(pred)).sum().item()

            eval_acc = 100. * correct_2 / n_samples_2

            print('\n')
            print('Clean model {} on dataset {} acc:{}'.format(model_i, args.dataset, clean_acc))
            print("Class_0_ASR:{}".format(overall_pred_target_acc_0))
            print("ASR of class-0 trigger on non-class-0 data:{}".format(non_target_attack_acc_0))  #
            print("Class_1_ASR:{}".format(overall_pred_target_acc_1))
            print("ASR of class-1 trigger on non-class-1 data:{}".format(non_target_attack_acc_1))  #
            print("Class_2_ASR:{}".format(overall_pred_target_acc_2))
            print("ASR of class-2 trigger on non-class-1 data:{}".format(non_target_attack_acc_2))  #

            print('Bkd model on benign prediction accuracy(BPA): %d/%d (%.2f%s) ' % (correct_2, n_samples_2, eval_acc, '%'))
            CAD = clean_acc - eval_acc
            print('CAD：{}'.format(CAD))

if __name__ == '__main__':
    args = parse_args()
    run(args)
