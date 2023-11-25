import torch
import torch.nn.functional as F
from pathlib import Path
import torch.distributed as dist
import torch.multiprocessing as mp

import time
from torch.utils.data import DataLoader
import os
from torch.optim import Adam


from steps.ingest_transform_data import data_loader
from steps.Model import Model
from steps.trainer import Trainer
from steps.config import CFG,conf


import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12350"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()

def cleanup():
    dist.destroy_process_group()
    
    
def main(rank: int, world_size: int,run,client):
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
    model = Model(name = "googlenet").get_model()
  
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)
    
    trainer = Trainer(model,conf, train_loader,test_loader , optimizer,rank,run)
    start = time.time()
    trainer.train(total_epochs)
    end = time.time()
    if(rank == 0):
        p_value = client.log_metric(run.info.run_id, "final_time", end - start)
    trainer.test(final_model_path)
    
    cleanup()
    
if __name__ == "__main__":
    
    client = MlflowClient(tracking_uri="http://0.0.0.0:5000")
    #exp_id = client.create_experiment("GoogleNet")
    exp_id = client.get_experiment_by_name("GoogleNet").experiment_id
    run_names = ['GPU:1','GPU:2','GPU:3','GPU:4']
    
    for i in range(0,4):
        
        run = client.create_run(exp_id, run_name=run_names[i])
    
        world_size = i+1

        p = client.log_param(run.info.run_id,"Num-GPUS", world_size) 
        p = client.log_param(run.info.run_id,"Num_Epochs", CFG.num_epochs) 
        p = client.log_param(run.info.run_id,"Batch_Size", CFG.batch_size) 
        p = client.log_param(run.info.run_id,"learning_rate", CFG.learning_rate) 
        
        #Multi_GPU training
        mp.spawn(main, args=(world_size,run,client), nprocs=world_size)