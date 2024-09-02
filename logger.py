import wandb
from datetime import datetime
import logging
from confidentials import WANDB_API_KEY
import os

def get_logger(name, log_dir='logs'):
    return Logger(name, log_dir)

class Logger:
    def __init__(self, name=None, log_dir='logs'):
        wandb.login(WANDB_API_KEY)
        if name is None:
            name = f"results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        self.run = wandb.init(project='fast-check',
                              name=name,)
        
        self.name = name
        logging.basicConfig(filename=os.path.join(log_dir, name, '.log'),
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
        self.logger = logging.getLogger(name)
    
    def print(self, string):
        print(string)
        self.logger.info(string)
    
    def log(self, **kwargs):
        self.run.log(kwargs)
        
    