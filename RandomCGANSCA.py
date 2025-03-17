from train_cgan import *
from create_models import *
#from compare_profiling_models import *


class RandomCGANSCA:

    def __init__(self, **kwargs):
        self.args = kwargs['args']
        self.models = CreateModels(self.args, random_hp=True)
        if  "main_path" in kwargs:
            self.main_path = kwargs["main_path"] 
        else:
            self.main_path = self.args["results_root_path"]

    def train_cgan(self):
        self.dir_results = create_directory_results(self.args, self.main_path)
        np.savez(f"{self.dir_results}/hp.npz", hp_d = self.models.hp_d, hp_g=self.models.hp_g)
        np.savez(f"{self.dir_results}/args.npz", args=self.args)
        train_cgan = TrainCGAN(self.args, self.models, self.dir_results)
        train_cgan.train()
        return train_cgan.nt_fake[-1]