from torch.utils.data import Dataset
import json
import os

def load_prompts(prompts_json):
    try:
        prompts = json.load(open(prompts_json, 'r'))
    except Exception as _:
        with open(prompts_json, 'r') as f:
            prompts = f.readlines()
        prompts = [json.loads(j.strip()) for j in prompts]
    jailbreaks = [j['content'] for j in jailbreaks]
    return prompts

class QueryDataset(Dataset):
    def __init__(self, jailbreaks_json, benign_json=None, name='jailbreaks'):
        self.queries = load_prompts(jailbreaks_json)
        self.labels = [1] * len(self.queries)
        
        if benign_json:
            self.queries += load_prompts(benign_json)
            self.labels += [0] * (len(self.queries) - len(self.labels))

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx], self.labels[idx]
    
def get_attack_json(model_name, attack_dir):
    return [x for x in os.listdir(attack_dir) if model_name in x][0]
    
def get_query_set_by_attack(model_name="llama2",
                  attack="GCG",
                  benign_json=None,):
    try:
        jailbreaks_json = get_attack_json(os.path.join(".", 'jailbreaks', attack.upper()))
    except Exception as e:
        raise Exception(f"Could not find attack json for {attack} in jailbreaks directory, error: {e}")
    return QueryDataset(jailbreaks_json, benign_json=benign_json, name=attack + (' & benign' if benign_json else ''))
    