import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from fastdef.evaluation import eval_defense
from fastdef.model import ChatModel
import argparse
import torch

attacks = ['AutoDAN', 'GCG', 'PAIR', 'TAP', 'RS']
defenses = ['self-defense', 'ladef'] # Add more defenses here

def main(**kwargs):
    
    if kwargs['defense'] == 'ladef':
        print("Warning: to use ladef, you have to load the model in lookahaed mode")
        kwargs['use_lade'] = True
    
    model = ChatModel(debug=kwargs['debug'],
                      model_name=kwargs['model_name'],
                      use_lade=kwargs['use_lade'],
                      )
    kwargs.pop('model_name')
    kwargs.pop('use_lade')
    
    eval_defense(model, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # LLM
    parser.add_argument(
        '--model_name',
        type=str,
        default='llama2',
        choices=['vicuna', 'llama2']
    )

    # Attack
    parser.add_argument(
        '--attack',
        type=str,
        default=None,
        choices=attacks
    )
    
    # Include Benign
    parser.add_argument(
        '--include_benign',
        type=bool,
        default=False,
    )
    
    # Defense
    parser.add_argument(
        '--defense',
        type=str,
        default='self-defense',
        choices=defenses,
    )
    
    # Debug Mode
    parser.add_argument(
        '--debug',
        type=bool,
        default=False,
    )

    # Do Logging
    parser.add_argument(
        '--do_log',
        type=bool,
        default=True,
    )

    # Using Lookahead decoding
    parser.add_argument(
        '--use_lade',
        type=bool,
        default=False,
    )
    
    main(**vars(parser.parse_args()))

