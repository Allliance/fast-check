from torch.utils.data import Dataset

class QueryDataset(Dataset):
    def __init__(self, queries, tokenizer, max_length=128):
        self.queries = queries
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        inputs = self.tokenizer(query, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
        return inputs