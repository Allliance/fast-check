import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from fastdef.evaluation import get_vanilla_asr
from fastdef.model import ChatModel
from fastdef.logger import get_logger
import argparse


attacks = ['AutoDAN', 'GCG', 'PAIR', 'TAP', 'RS']


def main(**kwargs):
    
    model = ChatModel(debug=kwargs['debug'],
                      model_name=kwargs['model_name'],
                      use_lade=kwargs['use_lade'],
                      )
    kwargs.pop('model_name')
    kwargs.pop('use_lade')
    
    get_vanilla_asr(model, **kwargs)


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
    
    main(**vars(parser.parse_args()))

