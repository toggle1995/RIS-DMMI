from .mask_predictor import Decoder
from .backbone import MultiModalSwinTransformer
from .backbone_resnet import MultiModalResNet
from ._utils import DMMI

__all__ = ['dmmi_swin', 'dmmi_res']

# DMMI based on swin-transformer
def _segm_dmmi_swin(pretrained, args):
    # initialize the SwinTransformer backbone with the specified version
    if args.swin_type == 'tiny':
        embed_dim = 96
        depths = [2, 2, 6, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'small':
        embed_dim = 96
        depths = [2, 2, 18, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'base':
        embed_dim = 128
        depths = [2, 2, 18, 2]
        num_heads = [4, 8, 16, 32]
    elif args.swin_type == 'large':
        embed_dim = 192
        depths = [2, 2, 18, 2]
        num_heads = [6, 12, 24, 48]
    else:
        assert False
    # args.window12 added for test.py because state_dict is loaded after model initialization
    if 'window12' in pretrained or args.window12:
        print('Window size 12!')
        window_size = 12
    else:
        window_size = 7

    if args.mha:
        mha = args.mha.split('-')  # if non-empty, then ['a', 'b', 'c', 'd']
        mha = [int(a) for a in mha]
    else:
        mha = [1, 1, 1, 1]

    out_indices = (0, 1, 2, 3)
    backbone = MultiModalSwinTransformer(embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                                         window_size=window_size,
                                         ape=False, drop_path_rate=0.3, patch_norm=True,
                                         out_indices=out_indices,
                                         use_checkpoint=False, num_heads_fusion=mha,
                                         fusion_drop=args.fusion_drop
                                         )
    if pretrained:
        print('Initializing Multi-modal Swin Transformer weights from ' + pretrained)
        backbone.init_weights(pretrained=pretrained)
    else:
        print('Randomly initialize Multi-modal Swin Transformer weights.')
        backbone.init_weights()


    model = DMMI(backbone, Decoder(8*embed_dim))

    return model

def _load_model_dmmi_swin(pretrained, args):
    model = _segm_dmmi_swin(pretrained, args)
    return model


def dmmi_swin(pretrained='', args=None):
    return _load_model_dmmi_swin(pretrained, args)


#############################################
# DMMI based on resnet

def _segm_dmmi_res(pretrained, args):
    backbone = MultiModalResNet(pretrained)

    model = DMMI(backbone, Decoder(2048))
    return model


def _load_model_dmmi_res(pretrained, args):
    model = _segm_dmmi_res(pretrained, args)
    return model


def dmmi_res(pretrained='', args=None):
    return _load_model_dmmi_res(pretrained, args)