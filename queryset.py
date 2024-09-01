from torch.utils.data import Dataset
import json

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