import torch
import torch.nn.functional as F
from pathlib import Path
import torch.distributed as dist
import torch.multiprocessing as mp


from torch.utils.data import DataLoader
import os
from torch.optim import Adam


from steps.ingest_transform_data import data_loader
from steps.Model import Model
from steps.trainer import Trainer
from steps.config import CFG,conf


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12351"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()

def cleanup():
    dist.destroy_process_group()
    
    
def main(rank: int, world_size: int):
    print(f"Running basic DDP example on rank {rank}.")
    ddp_setup(rank, world_size)
    
    total_epochs = CFG.num_epochs 
    batch_size = CFG.batch_size 
    learning_rate = CFG.learning_rate
    
    final_model_path = Path("CIFAR10_ddp_epoch199.pt")
       
    #Data
    train_loader = data_loader(data_dir='./data',batch_size=batch_size)
    test_loader = data_loader(data_dir='./data',batch_size=batch_size,test=True)
    
    #Model
    #--------------------------------------------------------------------------------------------------------------------
    model = Model(name = "VGG16").get_model()
  
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)
    
    trainer = Trainer(model,conf, train_loader,test_loader , optimizer,rank)
    #trainer.train(total_epochs)
    trainer.test(final_model_path)
    
    cleanup()
    
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)