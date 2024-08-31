from queryset import QuerySet
from utils import load_model_and_tokenizer


queryset_path = 'data/queries.json'

def evaluate(**kwargs):
    model, tokenizer = load_model_and_tokenizer(**kwargs)
    

if __name__ == '__main__':
    evaluate()