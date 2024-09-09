# from torch.utils.data import Dataset, DataLoader
import json
import os
import random
import pathlib
from copy import deepcopy

def load_prompts(prompts_json):
    try:
        prompts = json.load(open(prompts_json, 'r'))
    except Exception as _:
        with open(prompts_json, 'r') as f:
            prompts = f.readlines()
        prompts = [json.loads(j.strip()) for j in prompts]
    prompts = [j['content'] for j in prompts]
    return prompts

class QueryDataset:
    def __init__(self,
                 jailbreaks_json,
                 benign_json=None,
                 name='jailbreaks',
                 sample=False, **kwargs):
        self.name = name
        self.queries = load_prompts(jailbreaks_json)
        self.labels = [1] * len(self.queries)
        
        if benign_json:
            self.queries += load_prompts(benign_json)
            self.labels += [0] * (len(self.queries) - len(self.labels))

        if sample:
            indices = random.sample(range(len(self.queries)), min(20, len(self.queries)))
            self.queries = [self.queries[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

    def __str__(self) -> str:
        return f"Queryset Name: {self.name}\nTotal Number of queries: {len(self)}\nNumber of jailbreak queries: {sum(self.labels)}\nNumber of benign queries: {len(self) - sum(self.labels)}"

    def __len__(self):
        return len(self.queries)

    def subset(self, indices):
        instance = deepcopy(self)
        instance.queries = [instance.queries[i] for i in indices]
        instance.labels = [instance.labels[i] for i in indices]
        
        return instance

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        if isinstance(idx, list):
            return [self[i] for i in idx]
        
        return self.queries[idx], self.labels[idx]
    
def get_attack_json(model_name, attack_dir, attack_name):
    return os.path.join(attack_dir, [x for x in os.listdir(attack_dir) if model_name in x and x.startswith(attack_name.lower())][0])
    
def get_query_set_by_attack(model_name="llama2",
                  attack="GCG",
                  benign_json=None,
                  jailbreaks_root_dir=None,
                  batch_size=1,
                  **kwargs):

    if jailbreaks_root_dir is None:
        jailbreaks_root_dir = pathlib.Path(__file__).parent.resolve()
        
    if not os.path.exists(os.path.join(jailbreaks_root_dir, 'jailbreaks')):
        raise Exception("Could not find jailbreaks directory")
    try:
        jailbreaks_json = get_attack_json(model_name, os.path.join(jailbreaks_root_dir, 'jailbreaks', attack.upper()), attack)
    except Exception as e:
        raise Exception(f"Could not find attack json for {attack} in jailbreaks directory, error: {e}")
    queryset = QueryDataset(jailbreaks_json, benign_json=benign_json, name=attack + (' & benign' if benign_json else ''), **kwargs)
    
    return queryset
    