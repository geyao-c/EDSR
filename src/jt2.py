import time

import torch
from thop import profile
import utility
from option import args
from model import edsr, ecbedsr, plainecbedsr, vdsr, mdsr

def EDSRx2():
    args.model = 'EDSR'
    args.n_resblocks = 16
    args.n_feats = 64
    args.scale = [2]

    start = time.time()
    model1 = edsr.EDSR(args)
    # print(model1)
    img = torch.rand((1, 3, 640, 360))
    new_img = model1(img)
    print(new_img.shape)
    end = time.time()
    print('EDSRx2: ', round(end - start, 2))

    # img = torch.zeros((1, 3, 640, 640))
    # flops, params = profile(model1, inputs=(img,), verbose=True)
    # print('model1 params is {}, flops is {}'.format(params, flops))
    # return flops, params

def EDSRx3():
    args.model = 'EDSR'
    args.n_resblocks = 16
    args.n_feats = 64
    args.scale = [3]

    start = time.time()
    model1 = edsr.EDSR(args)
    # print(model1)
    img = torch.rand((1, 3, 426, 240))
    new_img = model1(img)
    print(new_img.shape)
    end = time.time()
    print('EDSRx3: ', round(end - start, 2))

    # img = torch.zeros((1, 3, 640, 640))
    # flops, params = profile(model1, inputs=(img,), verbose=True)
    # print('model1 params is {}, flops is {}'.format(params, flops))
    # return flops, params

def EDSRx4():
    args.model = 'EDSR'
    args.n_resblocks = 16
    args.n_feats = 64
    args.scale = [4]

    start = time.time()
    model1 = edsr.EDSR(args)
    # print(model1)
    img = torch.rand((1, 3, 320, 180))
    new_img = model1(img)
    print(new_img.shape)
    end = time.time()
    print('EDSRx4: ', round(end - start, 2))

    # img = torch.zeros((1, 3, 640, 640))
    # flops, params = profile(model1, inputs=(img,), verbose=True)
    # print('model1 params is {}, flops is {}'.format(params, flops))
    # return flops, params

def fun1():
    args.model = 'EDSR'
    args.n_resblocks = 16
    args.n_feats = 64
    args.scale = [4]

    model1 = edsr.EDSR(args)
    print(model1)
    img = torch.zeros((1, 3, 320, 180))
    new_img = model1(img)
    print(new_img.shape)

    img = torch.zeros((1, 3, 320, 180))
    flops, params = profile(model1, inputs=(img,), verbose=True)
    print('model1 params is {}, flops is {}'.format(params, flops))
    return flops, params

def fun2():
    args.model = 'ECBEDSR'
    args.n_resblocks = 16
    args.n_feats = 64
    args.group_num = 2

    model2 = ecbedsr.ECB_EDSR(args)
    # print(model2)
    img = torch.zeros((1, 3, 426, 240))
    flops, params = profile(model2, inputs=(img,), verbose=True)
    print('model2 params is {}, flops is {}'.format(params, flops))

def fun3():
    args.model = 'PLAINECBEDSR'
    args.n_resblocks = 20
    args.n_feats = 64
    args.group_num = 64
    args.scale = [2]

    start = time.time()
    model2 = plainecbedsr.Plain_ECB_EDSR(args)
    # print(model2)
    img = torch.rand((1, 3, 640, 360))  # x2
    # img = torch.zeros((1, 3, 320, 180))  # x4
    new_img = model2(img)
    print(new_img.shape)
    end = time.time()
    print('ECBEDSR x2', round(end - start, 2))

    # img = torch.zeros((1, 3, 640, 360))  # x2
    # img = torch.zeros((1, 3, 320, 180))  # x4
    # flops, params = profile(model2, inputs=(img,), verbose=True)
    # print('model2 params is {}, flops is {}'.format(params, flops))
    # return flops, params

def cal():
    flops1, params1 = fun1()
    flops2, params2 = fun3()
    print('model1 params1 is {}, flops1 is {}'.format(params1, flops1))
    print('model2 params2 is {}, flops2 is {}'.format(params2, flops2))
    print('params2 / params1 is {}, flops2 / flops1 is {}'.format(params2 / params1, flops2 / flops1))

def fun4():
    args.model = 'VDSR'
    args.n_resblocks = 20
    args.n_feats = 64

    model1 = vdsr.VDSR(args)
    print(model1)
    # print('Total params: %.2fM' % (sum(p.numel() for p in model1.parameters()) / 1000000.0))

    img = torch.zeros((1, 3, 1280, 720))
    new_img = model1(img)
    print(new_img.shape)

    img = torch.zeros((1, 3, 1280, 720))
    flops, params = profile(model1, inputs=(img,), verbose=True)
    print('model1 params is {}, flops is {}'.format(params, flops))

    return flops, params

def fun5():
    args.model = 'MDSR'
    args.n_resblocks = 80
    # args.n_feats = 64
    args.scale = [2]

    model1 = mdsr.MDSR(args)
    print(model1)
    # img = torch.zeros((1, 3, 320, 180))
    # new_img = model1(img)
    # print(new_img.shape)

    img = torch.zeros((1, 3, 320, 180))
    flops, params = profile(model1, inputs=(img,), verbose=True)
    print('model1 params is {}, flops is {}'.format(params, flops))
    return flops, params

def ECBEDSRx2():
    args.model = 'PLAINECBEDSR'
    args.n_resblocks = 20
    args.n_feats = 64
    args.group_num = 64
    args.scale = [2]

    start = time.time()
    model2 = plainecbedsr.Plain_ECB_EDSR(args)
    # print(model2)
    img = torch.rand((1, 3, 640, 360))  # x2
    # img = torch.zeros((1, 3, 320, 180))  # x4
    new_img = model2(img)
    print(new_img.shape)
    end = time.time()
    print('ECBEDSR x2', round(end - start, 2))

def ECBEDSRx3():
    args.model = 'PLAINECBEDSR'
    args.n_resblocks = 20
    args.n_feats = 64
    args.group_num = 64
    args.scale = [3]

    start = time.time()
    model2 = plainecbedsr.Plain_ECB_EDSR(args)
    # print(model2)
    img = torch.rand((1, 3, 426, 240))  # x2
    # img = torch.zeros((1, 3, 320, 180))  # x4
    new_img = model2(img)
    print(new_img.shape)
    end = time.time()
    print('ECBEDSR x3', round(end - start, 2))

def ECBEDSRx4():
    args.model = 'PLAINECBEDSR'
    args.n_resblocks = 20
    args.n_feats = 64
    args.group_num = 64
    args.scale = [4]

    start = time.time()
    model2 = plainecbedsr.Plain_ECB_EDSR(args)
    # print(model2)
    img = torch.rand((1, 3, 320, 180))  # x2
    # img = torch.zeros((1, 3, 320, 180))  # x4
    new_img = model2(img)
    print(new_img.shape)
    end = time.time()
    print('ECBEDSR x4', round(end - start, 2))

if __name__ == '__main__':
    # fun1()
    # fun2()
    EDSRx2()
    EDSRx3()
    EDSRx4()
    ECBEDSRx2()
    ECBEDSRx3()
    ECBEDSRx4()
    # fun3()
    # cal()
    # fun4()
    # fun5()

# model1 params is 1517571.0, flops is 4577292288.0
# model1 params is 1222147.0, flops is 2815826688.0
# 928003.0, flops is 215008358400.0