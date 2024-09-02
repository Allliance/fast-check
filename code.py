# %%
from evaluate import get_asr
from model import ChatModel

DEBUG = False

attacks = ['AutoDAN', 'GCG', 'PAIR', 'TAP', 'RS']
model = ChatModel(debug=DEBUG)
print(f"Total number of parameters {model.model.num_parameters()/1e9}B")

# %%
import torch
import gc
torch.cuda.empty_cache()
gc.collect()

# %%
from evaluate import get_asr

results, asr = get_asr(model,
                       attack=attacks[-3],
                       log=True,
                       debug=DEBUG,
                       safe_check_method='keyword')


