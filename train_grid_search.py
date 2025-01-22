import itertools
import os
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score

from utils import initialize_parameters, print_model_info, update_seed_metrics, update_run_metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import random
import numpy as np
from tqdm import tqdm
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch import nn
from read_data import read_mimic
# from torch.utils.tensorboard import SummaryWriter
import pdb
from models.pvig_gaze import pvig_ti_224_gelu, pvig_s_224_gelu, pvig_m_224_gelu, pvig_b_224_gelu, pvig_s_224_gelu_2


def train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()
    size = len(dataloader.dataset)
    pbar = tqdm(dataloader, total=int(len(dataloader)))
    count = 0
    train_loss = 0.0
    train_acc = 0.0
    for batch, sample in enumerate(pbar):
        x, labels, gaze = sample
        x, labels, gaze = x.to(device), labels.to(device), gaze.to(device)
        # with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
        #     outputs = model(x, gaze)
        #     loss = loss_fn(outputs, labels)  # labels.shape = (,batch_size)
        outputs = model(x, gaze)
        loss = loss_fn(outputs, labels)  # labels.shape = (,batch_size)
        _, pred = torch.max(outputs, 1)
        num_correct = (pred == labels).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        acc = num_correct.item() / len(labels)
        count += len(labels)
        train_loss += loss * len(labels)
        train_acc += num_correct.item()
        pbar.set_description(f"loss: {loss:>f}, acc: {acc:>f}, [{count:>d}/{size:>d}]")

    return train_loss / count, train_acc / count, model


def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)

    pbar = tqdm(dataloader, total=int(len(dataloader)))
    count = 0
    test_loss = 0.0
    test_acc = 0.0
    test_p = 0.0
    test_r = 0.0
    test_f1 = 0.0
    gt = []
    pd = []
    fpr = dict()
    tpr = dict()
    roc_auc = []
    n_classes = 2
    cont_batches = 0
    with torch.no_grad():
        for batch, sample in enumerate(pbar):
            cont_batches += 1
            x, labels, gaze = sample
            x, labels, gaze = x.to(device), labels.to(device), gaze.to(device)

            outputs = model(x, gaze)
            loss = loss_fn(outputs, labels)
            _, pred = torch.max(outputs, 1)
            num_correct = (pred == labels).sum()
            loss = loss.item()
            acc = num_correct.item() / len(labels)
            count += len(labels)
            test_loss += loss * len(labels)
            test_acc += num_correct.item()
            test_p += precision_score(labels.cpu(), pred.cpu(), zero_division=0)
            test_r += recall_score(labels.cpu(), pred.cpu(), zero_division=0)
            test_f1 += f1_score(labels.cpu(), pred.cpu(), zero_division=0)

            gt.extend(labels.cpu().numpy())
            pd.extend(outputs.cpu().numpy())

            pbar.set_description(f"loss: {loss:>f}, acc: {acc:>f}, [{count:>d}/{size:>d}]")

    gt = np.array(gt)
    pd = np.array(pd)
    gt = label_binarize(np.array(gt), classes=[0, 1])  # classes=[0, 1]

    fpr[0], tpr[0], _ = roc_curve(gt[:, 0], pd[:, 0])
    roc_auc.append(auc(fpr[0], tpr[0]))
    aucavg = np.mean(roc_auc)
    print("AUC: {}".format(roc_auc))

    return test_loss / count, test_acc / count, test_p / cont_batches, test_r / cont_batches, test_f1 / cont_batches, aucavg, model


if __name__ == '__main__':
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print('Device:', device)
    save_dir = r'output/grid_search_ckpt'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    data_dir = r'../mimic_part_jpg'
    # writer = SummaryWriter(save_dir)

    # Grid search parameters and space
    batch_size_list = [128, 64, 32]  # [64, 32]
    hidden_dim_list = [256, 128, 64]
    dropout_list = [0.2, 0.4, 0.6]
    pyramid_levels_list = [[2, 4, 6], [3, 6]]
    blocks_list = [[1, 1, 3, 1]]

    combinations = list(itertools.product(batch_size_list,
                                          hidden_dim_list,
                                          dropout_list,
                                          pyramid_levels_list,
                                          blocks_list))

    print(f'Total number of combinations: {len(combinations)}')
    all_run_cont = len(combinations)
    cont = -1
    seed_list = [14, 267]  # , 917, 271
    for run_combination in combinations:
        cont += 1
        if cont<13:
            continue
        grid_params_dict = initialize_parameters(cont, run_combination)

        # Reset experiments recorder list
        val_results = list()
        test_results = list()

        """ Step 0 - Parameters """
        for seed in seed_list:
            # Reproducibility
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # Update configuration
            batch_size_item = run_combination[0]
            hidden_dim_item = run_combination[1]
            dropout_item = run_combination[2]
            pyramid_levels_item = run_combination[3]
            blocks_item = run_combination[4]

            batchsize = batch_size_item
            n_epochs = 100
            Lr = 1e-4
            evaluate_train = True
            check_val_every = 2
            criterion = nn.CrossEntropyLoss().to(device)

            # Get data
            train_generator, val_generator, test_generator = read_mimic(batchsize, data_dir)

            # Get model
            model = pvig_s_224_gelu_2(dropout=dropout_item,
                                      emb_dims=hidden_dim_item,
                                      pyramid_levels=pyramid_levels_item,
                                      blocks=blocks_item, )
            model = model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=Lr)
            print_model_info(model)

            idx_best_loss = 0
            idx_best_acc = 0
            idx_best_auc = 0
            val_loss = 0.0
            val_acc = 0.0
            val_auc = 0.0
            val_p = 0.0
            val_r = 0.0
            val_f1 = 0.0

            log_train_loss = []
            log_train_acc = []
            log_val_loss = []
            log_val_acc = []
            log_auc = []
            log_val_p = []
            log_val_r = []
            log_val_f1 = []

            for epoch in range(1, n_epochs + 1):
                # Train step
                print("===> Epoch {}/{}, learning rate: {}".format(epoch, n_epochs, Lr))
                train_loss, train_acc, model = train_loop(train_generator, model, criterion, optimizer, device)

                # Train metrics
                log_train_loss.append(train_loss)
                log_train_acc.append(train_acc)

                # Val step
                if epoch % check_val_every == 0:
                    val_loss, val_acc, val_p, val_r, val_f1, val_auc, model = test_loop(val_generator,
                                                                                        model,
                                                                                        criterion,
                                                                                        device)
                    # Val metrics
                    log_val_loss.append(val_loss)
                    log_val_acc.append(val_acc)
                    log_val_p.append(val_p)
                    log_val_r.append(val_r)
                    log_val_f1.append(val_f1)
                    log_auc.append(val_auc)

                    if val_f1 >= log_val_f1[int(idx_best_loss/check_val_every)]:
                        print("Save f1-best model.")
                        torch.save(model.state_dict(), os.path.join(save_dir, 'model_ckpt.pth'))
                        idx_best_loss = epoch - 1

                print("Training loss: {:f}, acc: {:f}".format(train_loss, train_acc))
                print("Validation loss: {:f}, acc: {:f}, p: {:f}, r: {:f}, f1: {:f}, auc: {:f}".format(val_loss,
                                                                                                       val_acc,
                                                                                                       val_p,
                                                                                                       val_r,
                                                                                                       val_f1,
                                                                                                       val_auc))

            # Test step
            state_dict = torch.load(os.path.join(save_dir, 'model_ckpt.pth'))
            model.load_state_dict(state_dict)
            test_loss, test_acc, test_p, test_r, test_f1, test_auc, model = test_loop(test_generator,
                                                                                      model,
                                                                                      criterion,
                                                                                      device)

            # Seed results
            best_epoch = log_val_f1.index(max(log_val_f1))
            best_val_loss = log_val_loss[best_epoch]
            best_val_acc = log_val_acc[best_epoch]
            best_val_p = log_val_p[best_epoch]
            best_val_r = log_val_r[best_epoch]
            best_val_f1 = log_val_f1[best_epoch]
            best_val_auc = log_auc[best_epoch]
            seed_res_val = [best_val_loss, best_val_f1, best_val_p, best_val_r, best_val_acc, best_val_auc]
            seed_res_test = [test_loss, test_f1, test_p, test_r, test_acc, test_auc]
            val_results, test_results = update_seed_metrics(seed_res_val, seed_res_test, val_results, test_results)

        # Save run results
        update_run_metrics(val_results, test_results, grid_params_dict)
