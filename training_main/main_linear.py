from __future__ import print_function
from config.config_linear_competition import parse_option

from training_linear.training_one_epoch_ckpt_competition import main_multilabel_competition

if __name__ == '__main__':
    opt = parse_option()

    if (opt.competition == 1):
        main_multilabel_competition()