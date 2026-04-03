import torch


class FuXiConfig:
    def __init__(self):
        self.img_size = (2, 721, 1440)
        self.patch_size = (2, 4, 4)
        self.in_chans = 70
        self.embed_dim = 1536
        self.num_groups = 32
        self.input_resolution = (180, 360)
        self.num_heads = 8
        self.window_size = 7
        self.depth = 48
        self.out_channels = 70 * 4 * 4

        self.batch_size = 1
        self.learning_rate = 2.5e-4
        self.weight_decay = 0.1
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.drop_path_rate = 0.2
        self.num_epochs = 100
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
