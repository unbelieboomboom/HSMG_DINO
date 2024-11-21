_base_ = 'hsm_grounding_dino_hsmdecoderloss_swin-t_finetune_16xb2_1x_cityscapes.py'

data_root = 'G:/datasets/cityscapes_used/'
# class_name = ('cat', )
class_name = ('person', 'car', 'truck', 'rider', 'bicycle', 'motorcycle', 'bus', 'train',)
num_classes = len(class_name)
metainfo = dict(classes=class_name)

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'  # noqa

model = dict(
    num_queries=900,
    bbox_head=dict(num_classes=num_classes,
                   hsm_dict=dict(
                       v_dim=256,
                       hsm_prototype_num=8,
                   ),
                   loss_cls_hsm=dict(
                       type='FocalLoss',
                       use_sigmoid=True,
                       gamma=2.0,
                       alpha=0.25,
                       loss_weight=0.2),
                   cross_attn_hsm_cfg=dict(
                       embed_dims=256,
                       num_heads=8,
                       dropout=0.0,
                       batch_first=True),
                   hsm_loss_sign=False,
                   hsm_loss_ratio=0.0,
                   decoder_loss_sign=True,
                   get_hsm_indices_sign=True
                   )
)

# train_dataloader = dict(
#     dataset=dict(
#         data_root=data_root,
#         metainfo=metainfo,
#         ann_file='annotations/trainval.json',
#         data_prefix=dict(img='images/')))

train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            metainfo=metainfo)))

# val_dataloader = dict(
#     dataset=dict(
#         metainfo=metainfo,
#         data_root=data_root,
#         ann_file='annotations/test.json',
#         data_prefix=dict(img='images/')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo))

test_dataloader = val_dataloader

# val_evaluator = dict(ann_file=data_root + 'annotations/test.json')
# test_evaluator = val_evaluator

max_epoch = 12

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=1))
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=30),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[11],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(lr=0.00005),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0),
        }))

# auto_scale_lr = dict(base_batch_size=16)
auto_scale_lr = dict(enable=True, base_batch_size=4)
