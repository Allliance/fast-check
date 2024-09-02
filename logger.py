import wandb
from datetime import datetime
# import logging
from fastdef.confidentials import WANDB_API_KEY
import os
from pathlib import Path

def get_logger(name, log_dir='logs'):
    return Logger(name, log_dir)

class Logger:
    def __init__(self, name=None, log_dir='logs'):
        wandb.login(anonymous="allow", key=WANDB_API_KEY)
        
        if name is None:
            name = f"results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        self.run = wandb.init(project='fastdef',
                              name=name,
                              dir=log_dir,)
    
    def log(self, **kwargs):
        try:
            self.run.log(kwargs)
        except Exception as e:
            print("Error loggin to wandb:", str(e))
        
    