from fastdef.evaluation import get_asr
from fastdef.model import ChatModel
import argparse
import os, sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

attacks = ['AutoDAN', 'GCG', 'PAIR', 'TAP', 'RS']


def main(**kwargs):
    
    model = ChatModel(debug=kwargs['debug'],
                      model_name=kwargs['model'],
                      use_lade=kwargs['use_lade'],
                      )
    
    get_asr(model, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # LLM
    parser.add_argument(
        '--model',
        type=str,
        default='llama2',
        choices=['vicuna', 'llama2']
    )

    # Attack
    parser.add_argument(
        '--attack',
        type=str,
        default='GCG',
        choices=attacks
    )
    
    # Attack
    parser.add_argument(
        '--debug',
        type=bool,
        default=False,
    )

    # Attack
    parser.add_argument(
        '--safe_check_method',
        type=str,
        default='keyword',
        choices=['keyword', 'model_based'],
    )
    
    # Do Logging
    parser.add_argument(
        '--do_log',
        type=bool,
        default=True,
    )

    # Do Logging
    parser.add_argument(
        '--use_lade',
        type=bool,
        default=False,
    )

    main(**parser.parse_args())

