import os.path

import torch
import numpy as np

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import sys
import argparse
from model import edsr
sys.path.append('')

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def get_baseline_model(args):
    origin_model = edsr.EDSR(args)

    model_path = '../model_pytorch/edsr_baseline_x2.pt'
    ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    origin_model.load_state_dict(ckpt, strict=True)

    return origin_model

def _load_weight(ci_index, last_select_index, state_dict, oristate_dict, conv_weight_name, conv_bias_name, prefix='model.'):
    oriweight = oristate_dict[conv_weight_name]
    curweight = state_dict[prefix + conv_weight_name]

    orifilter_num = oriweight.size(0)
    currentfilter_num = curweight.size(0)
    ci_dir = '../edsr_x2_l1_ci'
    print(orifilter_num, currentfilter_num)

    if orifilter_num != currentfilter_num:
        ci_path = os.path.join(ci_dir, 'ci_conv{}.npy'.format(ci_index))
        ci = np.load(ci_path)
        print('loading ci from: {}'.format(ci_path))
        select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
        select_index.sort()

        if last_select_index is not None:
            for index_i, i in enumerate(select_index):
                for index_j, j in enumerate(last_select_index):
                    state_dict[prefix + conv_weight_name][index_i][index_j] = oristate_dict[conv_weight_name][i][j]
        else:
            for index_i, i in enumerate(select_index):
                state_dict[prefix + conv_weight_name][index_i] = oristate_dict[conv_weight_name][i]

        # 加载bias层
        for index_i, i in enumerate(select_index):
            state_dict[prefix + conv_bias_name][index_i] = oristate_dict[conv_bias_name][i]

        last_select_index = select_index

    elif last_select_index is not None:
        for index_i in range(orifilter_num):
            for index_j, j in enumerate(last_select_index):
                state_dict[prefix + conv_weight_name][index_i][index_j] =  oristate_dict[conv_weight_name][index_i][j]

        # 加载bias层
        state_dict[prefix + conv_bias_name] = oristate_dict[conv_bias_name]

        last_select_index = None
    else:
        # logger.info('yes yes')
        state_dict[prefix + conv_weight_name] = oriweight

        # 加载bias层
        state_dict[prefix + conv_bias_name] = oristate_dict[conv_bias_name]

        last_select_index = None

    return last_select_index

def load_pruned_edsr(model, oristate_dict):
    last_select_index = None
    state_dict = model.state_dict()
    ci_index = 1
    # 裁剪第一层
    conv_name = 'head.0'
    conv_weight_name = conv_name + '.weight'
    conv_bias_name = conv_name + '.bias'
    last_select_index = _load_weight(ci_index, last_select_index, state_dict, oristate_dict, conv_weight_name, conv_bias_name)
    ci_index += 1

    # 裁剪body
    for i in range(16):
        conv_name = 'body.{}'.format(i)
        conv_weight_name = conv_name + '.body.0' + '.weight'
        conv_bias_name = conv_name + '.body.0' + '.bias'
        last_select_index = _load_weight(ci_index, last_select_index, state_dict, oristate_dict, conv_weight_name,
                                         conv_bias_name)
        ci_index += 1

        conv_name = 'body.{}'.format(i)
        conv_weight_name = conv_name + '.body.2' + '.weight'
        conv_bias_name = conv_name + '.body.2' + '.bias'
        last_select_index = _load_weight(ci_index, last_select_index, state_dict, oristate_dict, conv_weight_name,
                                         conv_bias_name)
        ci_index += 1

    # 裁剪body的第16层
    conv_name = 'body.16'
    conv_weight_name = conv_name + '.weight'
    conv_bias_name = conv_name + '.bias'
    last_select_index = _load_weight(ci_index, last_select_index, state_dict, oristate_dict, conv_weight_name, conv_bias_name)
    ci_index += 1

    # 裁剪tail
    conv_name = 'tail.0.0'
    conv_weight_name = conv_name + '.weight'
    conv_bias_name = conv_name + '.bias'
    last_select_index = _load_weight(ci_index, last_select_index, state_dict, oristate_dict, conv_weight_name, conv_bias_name)
    ci_index += 1

    conv_name = 'tail.1'
    conv_weight_name = conv_name + '.weight'
    conv_bias_name = conv_name + '.bias'
    last_select_index = _load_weight(ci_index, last_select_index, state_dict, oristate_dict, conv_weight_name, conv_bias_name)
    ci_index += 1

    model.load_state_dict(state_dict)

def main():
    global model
    if checkpoint.ok:
        loader = data.Data(args)
        _model = model.Model(args, checkpoint)
        print(_model)
        args.pruning_ratio = 1.0
        origin_model = get_baseline_model(args)
        load_pruned_edsr(_model, origin_model.state_dict())
        _loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, _model, _loss, checkpoint)
        while not t.terminate():
            t.train()
            t.test()
        checkpoint.done()

if __name__ == '__main__':
    # get_baseline_model()
    main()
