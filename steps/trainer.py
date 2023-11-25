import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torchmetrics
import torch.nn as nn

from steps.config import CFG

#Tracking metrics
import time
import mlflow
from mlflow.tracking import MlflowClient

class Trainer:
    def __init__(
        self,
        model,
        conf,
        train_data: DataLoader,
        test_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        run,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_loader = train_data
        self.test_loader = test_data
        self.optimizer = optimizer
        self.model = DDP(model, device_ids=[gpu_id])
        self.conf = conf
        self.run = run
        self.client = MlflowClient(tracking_uri="http://0.0.0.0:5000")
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=CFG.num_classes, average="micro"
        ).to(self.gpu_id)

        self.valid_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=100, average="micro"
        ).to(self.gpu_id)
        
        self.correct = 0
        self.total = 0

    def _run_batch(self, src, tgt):
        
        self.optimizer.zero_grad()
        
        out = self.model(src)
        _, predicted = torch.max(out.data, 1)
        #print(predicted.shape)
        self.correct += (predicted == tgt).float().sum()
        
        loss = self.criterion(out, tgt)
        loss.backward()
        self.optimizer.step()
        
        # self.train_acc.update(out, tgt)
        return loss.item()

    def _run_epoch(self, epoch):
        self.train_loader.sampler.set_epoch(epoch)
        loss = 0.0
        self.correct = 0
        self.total = 0
        for src, tgt in self.train_loader:
            src = src.to(self.gpu_id)
            tgt = tgt.to(self.gpu_id)
            self.total = self.total + src.size(0)
           # print(self.total)
            loss_batch = self._run_batch(src, tgt)
            loss += loss_batch
         
        if(epoch % self.conf["save_every"] == 0):
            # Mlflow Logging
            if(self.gpu_id == 0):
                with torch.no_grad():
                    p_value = self.client.log_metric(self.run.info.run_id, "loss", loss ,step=epoch)
                    p_value = self.client.log_metric(self.run.info.run_id, "train_acc", 100 * self.correct / self.total ,step=epoch)
            print(f"GPU {self.gpu_id} Epoch : {epoch} Loss : {loss}")
            
              
    def _save_checkpoint(self, epoch: int):
        ckp = self.model.state_dict()
        model_path = self.conf["trained_models"] + f"CIFAR10_ddp_epoch{epoch}.pt"
        torch.save(ckp, model_path)

    def train(self, max_epochs: int):
        self.model.train()
        for epoch in range(max_epochs):
            start = time.time()
            self._run_epoch(epoch)
            end = time.time()
            if self.gpu_id == 0 and epoch % self.conf["save_every"] == 0:
                p_value = self.client.log_metric(self.run.info.run_id, "time", end - start ,step=epoch)
                
                #self._save_checkpoint(epoch)
        # save last epoch
        #self._save_checkpoint(max_epochs - 1)
        
    def test(self, final_model_path: str):
        #self.model.load_state_dict(
        #    torch.load(final_model_path, map_location="cpu")
        # )
        self.model.eval()

        with torch.no_grad():
            for src, tgt in self.test_loader:
                src = src.to(self.gpu_id)
                tgt = tgt.to(self.gpu_id)
                out = self.model(src)
                self.valid_acc.update(out, tgt)
            p_value = self.client.log_metric(self.run.info.run_id, "final_test_Acc", 100 * self.valid_acc.compute().item() )
        
        print(
            f"[GPU{self.gpu_id}] Test Acc: {100 * self.valid_acc.compute().item():.4f}%"
        )
