from queryset import QuerySet
from utils import load_model_and_tokenizer

attacks = ['AutoDAN', 'GCG', 'PAIR', 'TAP', 'RS']
defenses = ['Self-Defense'] # Add more defenses here
model_names = ['llama', 'vicuna']

def get_asr(model, attack, defense=None):
    
    pass

def evaluate(model_name=model_names[0],
             attack=attacks[0],
             defense=defenses[0],
             **kwargs):
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        **kwargs)
    


if __name__ == '__main__':
    evaluate()