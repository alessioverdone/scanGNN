import torch


def initialize_parameters(cont, run_combination):
    batch_size_item = run_combination[0]
    hidden_dim_item = run_combination[1]
    dropout_item = run_combination[2]
    pyramid_levels_item = run_combination[3]
    blocks_item = run_combination[4]

    grid_params_name = ['Run',
                        'batch_size',
                        'hidden_dim',
                        'dropout',
                        'pyramid_levels',
                        'blocks',
                        'val_loss_mean',
                        'val_loss_std',
                        'val_acc_mean',
                        'val_acc_std',
                        'val_p_mean',
                        'val_p_std',
                        'val_r_mean',
                        'val_r_std',
                        'val_f1_mean',
                        'val_f1_std',
                        'val_auroc_mean',
                        'val_auroc_std',
                        'test_loss_mean',
                        'test_loss_std',
                        'test_acc_mean',
                        'test_acc_std',
                        'test_p_mean',
                        'test_p_std',
                        'test_r_mean',
                        'test_r_std',
                        'test_f1_mean',
                        'test_f1_std',
                        'test_auroc_mean',
                        'test_auroc_std']
    grid_params = [cont,
                   batch_size_item,
                   hidden_dim_item,
                   dropout_item,
                   pyramid_levels_item,
                   blocks_item,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.]
    grid_params_dict = dict(zip(grid_params_name, grid_params))
    output_string = ' '.join([f'{name}: {value}' for name, value in grid_params_dict.items()])
    print(output_string)
    return grid_params_dict


def print_model_info(model):
    print(model)
    print("There are", sum(p.numel() for p in model.parameters()), "parameters.")
    print("There are", sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable parameters.")
    if torch.cuda.is_available(): print('Using CUDA!')


def update_seed_metrics(seed_res_val, seed_res_test, val_results, test_results):
    best_val_loss, best_val_f1, best_val_p, best_val_r, best_val_acc, best_val_auroc = seed_res_val

    # Testing
    test_loss, test_f1, test_p, test_r, test_acc, test_auroc = seed_res_test

    val_results.append([best_val_loss, best_val_f1, best_val_p, best_val_r, best_val_acc, best_val_auroc])
    test_results.append([test_loss, test_f1, test_p, test_r, test_acc, test_auroc])

    print(f'best_val_loss: {best_val_loss}')
    print(f'best_val_f1: {best_val_f1}')
    print(f'best_val_p: {best_val_p}')
    print(f'best_val_r: {best_val_r}')
    print(f'best_val_acc: {best_val_acc}')
    print(f'best_val_auroc: {best_val_auroc}')
    print(f'test_loss: {test_loss}')
    print(f'test_f1: {test_f1}')
    print(f'test_p: {test_p}')
    print(f'test_r: {test_r}')
    print(f'test_acc: {test_acc}')
    print(f'test_auroc: {test_auroc}')

    return val_results, test_results


def update_run_metrics(val_results, test_results, grid_params_dict, save_logs=True):
    val_results = torch.tensor(val_results)
    test_results = torch.tensor(test_results)
    val_loss_over_seeds = val_results[:, 0]
    val_f1_over_seeds = val_results[:, 1]
    val_p_over_seeds = val_results[:, 2]
    val_r_over_seeds = val_results[:, 3]
    val_acc_over_seeds = val_results[:, 4]
    val_auroc_over_seeds = val_results[:, 5]

    test_loss_over_seeds = test_results[:, 0]
    test_f1_over_seeds = test_results[:, 1]
    test_p_over_seeds = test_results[:, 2]
    test_r_over_seeds = test_results[:, 3]
    test_acc_over_seeds = test_results[:, 4]
    test_auroc_over_seeds = test_results[:, 5]

    grid_params_dict.update({
        'val_loss_mean': float(torch.mean(val_loss_over_seeds)),
        'val_loss_std': float(torch.std(val_loss_over_seeds)),
        'val_f1_mean': float(torch.mean(val_f1_over_seeds)),
        'val_f1_std': float(torch.std(val_f1_over_seeds)),
        'val_p_mean': float(torch.mean(val_p_over_seeds)),
        'val_p_std': float(torch.std(val_p_over_seeds)),
        'val_r_mean': float(torch.mean(val_r_over_seeds)),
        'val_r_std': float(torch.std(val_r_over_seeds)),
        'val_acc_mean': float(torch.mean(val_acc_over_seeds)),
        'val_acc_std': float(torch.std(val_acc_over_seeds)),
        'val_auroc_mean': float(torch.mean(val_auroc_over_seeds)),
        'val_auroc_std': float(torch.std(val_auroc_over_seeds)),
        'test_loss_mean': float(torch.mean(test_loss_over_seeds)),
        'test_loss_std': float(torch.std(test_loss_over_seeds)),
        'test_f1_mean': float(torch.mean(test_f1_over_seeds)),
        'test_f1_std': float(torch.std(test_f1_over_seeds)),
        'test_p_mean': float(torch.mean(test_p_over_seeds)),
        'test_p_std': float(torch.std(test_p_over_seeds)),
        'test_r_mean': float(torch.mean(test_r_over_seeds)),
        'test_r_std': float(torch.std(test_r_over_seeds)),
        'test_acc_mean': float(torch.mean(test_acc_over_seeds)),
        'test_acc_std': float(torch.std(test_acc_over_seeds)),
        'test_auroc_mean': float(torch.mean(test_auroc_over_seeds)),
        'test_auroc_std': float(torch.std(test_auroc_over_seeds))
    })
    output_string = ' '.join([f'{name}: {value}' for name, value in grid_params_dict.items()])

    if save_logs:
        with open(f'logs/logs_gazeGNN_mean_std.txt', 'a') as file:
            print(output_string, file=file)