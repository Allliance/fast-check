# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from fastdef.queryset import get_query_set_by_attack
from fastdef.utils import load_model_and_tokenizer
from fastdef.safety_checker import JBChecker
from fastdef.defenses.utils import get_defense
import torch, gc
from fastdef.logger import get_logger

ATTACK = ['AutoDAN', 'GCG', 'PAIR', 'TAP', 'RS'][-1]
defenses = ['Self-Defense', 'fastdef'] # Add more defenses here
model_names = ['llama', 'vicuna']

def get_vanilla_asr(model,
                    attack='RS',
                    safe_check_method='keyword',
                    log=True,
                    log_wandb=True,
                    debug=False,
                    logger=None,
                    **kwargs):
    
    if log_wandb and logger is None:
        logger = get_logger(f'{model.name}-ASR-{attack}' + ("DEBUG" if debug else ""),
                            debug=debug,
                            notes='Checking vanilla asr of the given jailbreaks' + f'on {model.name} model under {attack} attack',
                            config={
                                'model': model.name,
                                'attack': attack,
                                'safety_check_method': safe_check_method,
                                'log': log,
                                }
                            )
        # print = logger.print
    
    queryset = get_query_set_by_attack(model_name=model.name,
                                       attack=attack,
                                       sample=debug,
                                       **kwargs)
    if len(queryset) != sum(queryset.labels):
        print("Warning: The query set has benign queries")
        print("Removing the benign queries")
        mal_indices = [i for i, l in enumerate(queryset.labels) if l]
        
        queryset = queryset.subset(mal_indices)
    
    print(queryset)
    print("#" * 50)
    results = []
    
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
            new_results = [{'is_jailbreak': jailbroken, **r} for jailbroken, r in zip(jb_results, new_results)]
            
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
        
            
    asr = sum([r['is_jailbreak'] for r in results]) / sum(queryset.labels)
    
    if logger is not None:
        logger.log(ASR=asr)
    
    print("Attack Success Rate: ", asr)
    return results, asr


def eval_defense(model,
                attack='RS',
                defense='Self-Defense',
                benign_json_path=None,
                log=True,
                log_wandb=True,
                debug=False,
                logger=None,
                **kwargs):
    if log_wandb and logger is None:
        logger = get_logger(f'{model.name} {attack} X {defense}' + ("DEBUG" if debug else ""),
                            debug=debug,
                            notes='Evaluating the defense' + f'{defense} on {model.name} model under {attack} attack',
                            config={
                                'model': model.name,
                                'attack': attack,
                                'defense': defense,
                                'benign_json': benign_json_path,
                                'log': log,
                                }
                            )
    
    queryset = get_query_set_by_attack(model_name=model.name,
                                       attack=attack,
                                       sample=debug,
                                       benign_json=benign_json_path,
                                       **kwargs)
    
    print(queryset)
    print("#" * 50)
    results = []
    
    defense = get_defense(defense, model, **kwargs)
    
    for query, label in queryset:
        queries = [query]
        labels = [label]
        print("Working on a batch of size:", len(queries))
        
        is_jailbreaks = defense.is_jailbreak(queries)
        results.extend([{'is_jailbreak': i, 'prompt': q, 'label': l} for i, q, l in zip(is_jailbreaks, queries, labels)])
        
        print("A batch finished")
        for query, label, is_jailbreak in zip(queries, labels, is_jailbreaks):
            print("Prompt:", query)
            if label and is_jailbreak:
                print("Jailbreak Detected Successfully!")
            elif label and not is_jailbreak:
                print("Jailbreak Not Detected!")
            elif not label and is_jailbreak:
                print("False Positive!")
            else:
                print("True Negative!")
            print("-" * 50)
            print("-" * 50)
    
    metrics = {}
    if sum(queryset.labels) > 0:
        metrics['TPR (ASR)'] = sum([r['is_jailbreak'] for r in results]) / sum(queryset.labels)
    if sum(queryset.labels) < len(queryset):
        metrics['FPR'] = sum([r['is_jailbreak'] for r in results]) / (len(queryset) - sum(queryset.labels))

    for k, v in metrics.items():
        print(k, ":", v)
        
    return results, metrics
