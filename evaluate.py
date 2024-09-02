from fastdef.queryset import get_query_set_by_attack
from fastdef.utils import load_model_and_tokenizer
from fastdef.safety_checker import JBChecker
from fastdef.defenses.utils import get_defense
import torch, gc
from fastdef.logger import get_logger

attacks = ['AutoDAN', 'GCG', 'PAIR', 'TAP', 'RS']
defenses = ['Self-Defense'] # Add more defenses here
model_names = ['llama', 'vicuna']

def get_asr(model,
            attack='TAP',
            defense=None,
            safe_check_method='keyword',
            log=True,
            debug=False,
            **kwargs):
    if log:
        logger = get_logger(f'ASR-{attack}')
        # print = logger.print
    
    queryset = get_query_set_by_attack(model_name=model.name,
                                       attack=attack,
                                       sample=debug,
                                       **kwargs)
    print(queryset)
    print("#" * 50)
    results = []
    
    if defense is None:
        
        conversations = []
        for query, label in queryset:
            
            print("Step:", len(results) + 1)
            if log:
                print(f"Prompt: {query}\n")
            queries = [query]
            labels = [label]
            
            if safe_check_method == 'keyword':
                responses = model.chat(queries, max_new_tokens=100)
            else:
                responses = model.chat(queries)
            
            new_results = [{'prompt': q, 'label': l} for q, l in zip(queries, labels)]
            new_conversations = [{'response': r, 'prompt': q} for r, q in zip(responses, queries)]
            
            if safe_check_method == 'keyword':
                jb_checker = JBChecker(method=safe_check_method)
                jb_results = jb_checker.check(new_conversations)
                new_results = [{'is_jailbreak': not jailbroken, **r} for jailbroken, r in zip(jb_results, new_results)]
                
                if log:
                    for response, jailbroken in zip(responses, jb_results):
                        print(f"Response: {response}\n")
                        print("Is Jailbroken?", "Yes" if jailbroken else "No")
                    print("-" * 50)
                    print("-" * 50)

                    
            conversations.extend(new_conversations)
            results.extend(new_results)
        
        if safe_check_method == 'model_based':
            del model
            torch.cuda.empty_cache()
            gc.collect()
            
            # safety check

            jb_checker = JBChecker(method=safe_check_method)
            jailbroken_results = jb_checker.check(conversations)
            
            results = [{'is_jailbreak': not jailbroken, **r} for jailbroken, r in zip(jailbroken_results, results)]
    else:
        defense = get_defense(defense, model, **kwargs)
        for query, label in queryset:
            queries = [query]
            labels = [label]
            is_jailbreaks = defense.is_jailbreak(queries)
            results.extend([{'is_jailbreak': i, 'prompt': q, 'label': l} for i, q, l in zip(is_jailbreaks, queries, labels)])
            
    asr = sum([r['is_jailbreak'] for r in results]) / sum(queryset.labels)
    if log:
        print("Attack Success Rate: ", asr)
    return results, asr

    
    

def evaluate(model_name=model_names[0],
             attack=attacks[0],
             defense=defenses[0],
             **kwargs):
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        **kwargs)
    


if __name__ == '__main__':
    evaluate()