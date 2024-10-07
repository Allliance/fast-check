# from torch.utils.data import Dataset, DataLoader
import json
import os
import random
import pathlib
from copy import deepcopy

DEFAULT_ASSETS_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), 'assets')
DEFAULT_BENIGN_JSON = os.path.join(DEFAULT_ASSETS_DIR, 'benign_prompts.json')

def load_prompts(prompts_json):
    try:
        prompts = json.load(open(prompts_json, 'r'))
    except Exception as _:
        with open(prompts_json, 'r') as f:
            prompts = f.readlines()
        prompts = [json.loads(p.strip()) for p in prompts]
    prompts = [p['content'] for p in prompts]
    return prompts

class QueryDataset:
    def __init__(self,
                 jailbreaks_json_file=None,
                 benigns_json_file=None,
                 name='',
                 sample=False,
                 balanced=True,
                 random_seed=42,
                 **kwargs):
        assert benigns_json_file is not None or jailbreaks_json_file is not None
        
        self.name = name
        self.queries = load_prompts(jailbreaks_json_file)
        self.labels = [1] * len(self.queries)
        
        if benigns_json_file is not None:
            new_queries = load_prompts(benigns_json_file)
            self.queries += new_queries
            self.labels += [0] * len(new_queries)

        if balanced:
            if random_seed:
                random.seed(random_seed)
            min_len = min(len(self.queries), sum(self.labels))
            jailbreak_indices = random.sample([i for i, l in enumerate(self.labels) if l == 1], min_len)
            benign_indices = random.sample([i for i, l in enumerate(self.labels) if l == 0], min_len)
            
            self.queries = [self.queries[i] for i in jailbreak_indices] + [self.queries[i] for i in benign_indices]
            self.labels = [1] * min_len + [0] * min_len

        if sample:
            if random_seed:
                random.seed(random_seed)
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
    
def get_json_file(model_name, attack_dir, attack_name):
    return os.path.join(attack_dir, [x for x in os.listdir(attack_dir) if model_name in x and x.startswith(attack_name.lower())][0])
    
def get_query_set_by_attack(model_name="llama2",
                  attack=None,
                  include_benign=False,
                  assets_root_dir=None,
                  batch_size=1,
                  **kwargs):
    
    if assets_root_dir is None:
        assets_root_dir = DEFAULT_ASSETS_DIR
    
    if not os.path.exists(os.path.join(assets_root_dir, 'jailbreaks')):
        raise Exception("Could not find jailbreaks directory")
    
    jailbreaks_json_file = None
    benigns_json_file = None
    
    try:
        jailbreaks_json_file = get_json_file(model_name, os.path.join(assets_root_dir, 'jailbreaks', attack.upper()), attack)
    except Exception as e:
        raise Exception(f"Could not load json file for {attack}, error: {e}")
    
    if include_benign:
        benigns_json_file = os.path.join(assets_root_dir, 'benign_prompts.json')
        if not os.path.exists(benigns_json_file):
            raise Exception("Could not find benign prompts json file")
    
    
    queryset = QueryDataset(jailbreaks_json_file,
                            benigns_json_file,
                            name=attack + (' & benign' if include_benign else ''),
                            **kwargs)
    
    return queryset
    