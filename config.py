import torch


class Configuration():
    # Dataset parameters
    zones = ['FA', 'FB', 'FC', 'FD', 'FE', 'FF', 'FG', 'FH', 'FI']
    zones_operative = ['FA', 'FB', 'FC', 'FD', 'FE', 'FF', 'FG', 'FH', 'FI']
    dataset_name = 'd2'
    sites = ['parco_appia']
    zones_per_site = {'parco_appia': ['FA', 'FB', 'FC', 'FD', 'FE', 'FF', 'FG', 'FH', 'FI']}

    # Preprocessing
    subimages_overlap = 48
    subimages_image_size = 64
    use_only_anom_data = True

    # Training parameters
    max_epochs = 150
    batch_size = 32
    lr = 1e-4
    LR_PATIENCE = 10
    DECAY_LR = 0.9
    EARLY_STOP_PATIENCE = 10
    train_valtest_ratio = 0.8
    model_name = 'ViT'  # ['uNet', 'ViT']

    # Trainer parameters
    seed = 420  # torch.Generator().manual_seed(42)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = "auto"
    enable_checkpointing = True
    reproducible = True
    fast_dev_run = False
    enable_autocast = False
    verbose = True
    logger = None
    log_every_n_steps = 50
    limit_train_batches = 1.0
    limit_val_batches = 1.0
    check_val_every_n_epoch = 3
    len_val = 0
    len_test = 0
    test_step = False

    # Path parameters
    ckpt_dir = "./checkpoints"
    logs_dir = "../logs"
    directory_scanpath_ASD = r'C:\Users\Grid\Desktop\PhD\IJCNN\data\single_final_1010\scanpath_ASD'
    directory_img_ASD = r'C:\Users\Grid\Desktop\PhD\IJCNN\data\single_final_1010\images_ASD'
    directory_fixpoints_ASD = r'C:\Users\Grid\Desktop\PhD\IJCNN\data\single_final_1010\fix_maps_ASD'
    directory_scanpath_TD = r'C:\Users\Grid\Desktop\PhD\IJCNN\data\single_final_1010\scanpath_TD'
    directory_img_TD = r'C:\Users\Grid\Desktop\PhD\IJCNN\data\single_final_1010\images_TD'
    directory_fixMaps_TD = r'C:\Users\Grid\Desktop\PhD\IJCNN\data\single_final_1010\fix_maps_TD'

    # ViT Parameters
    hidden_size = 64
    image_size = 64
    patch_size = 8
    dim = 256
    depth = 8
    heads = 8
    mlp_dim = 64
    dropout = 0.15
    emb_dropout = 0.1

    # Inference
    zones_per_site_inference = {'parco_appia': ['FI']}
    inference_batch_size = 32
    # inference_overlap = 0
