import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn

from functools import reduce
import operator
from bert.modeling_bert import BertModel

from lib import segmentation
import pdb
import transforms 
from transforms import transform
from data.dataset_zom import Refzom_DistributedSampler,Referzom_Dataset
from data.dataset import ReferDataset
import utils
import numpy as np

import gc


def get_dataset(image_set, transform, args, eval_mode):
    if args.dataset == 'ref-zom':
        ds = Referzom_Dataset(args,
                    split=image_set,
                    image_transforms=transform,
                    target_transforms=None,
                    eval_mode=eval_mode
                    )
    else:
        ds = ReferDataset(args,
                        split=image_set,
                        image_transforms=transform,
                        target_transforms=None,
                        eval_mode=eval_mode
                        )
    num_classes = 2

    return ds, num_classes



def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U



def get_transform(args):
    transform = [transforms.Resize(args.img_size, args.img_size),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return transforms.Compose(transform)


def criterion(input, target):
    weight = torch.FloatTensor([0.9, 1.1]).cuda()
    return nn.functional.cross_entropy(input, target, weight=weight)


def evaluate(model, data_loader, bert_model):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_its = 0
    acc_ious = 0

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    mean_acc = []
    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            total_its += 1
            image, target, source_type, sentences, sentences1, attentions = data
            image, sentences, sentences1, attentions = image.cuda(non_blocking=True), \
                                                        sentences.cuda(non_blocking=True), \
                                                        sentences1.cuda(non_blocking=True), \
                                                        attentions.cuda(non_blocking=True)
            sentences = sentences.squeeze(1)
            sentences1 = sentences1.squeeze(1)
            attentions = attentions.squeeze(1)
            target = target.data.numpy()

            for j in range(sentences.size(-1)):

                last_hidden_states = bert_model(sentences[:, :, j], attention_mask=attentions[:, :, j])[0]
                embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
                embedding1 = embedding
                loss_contra, loss_lansim, output = model(image, embedding, embedding1, l_mask=attentions[:, :, j].unsqueeze(-1), training_flag=True)

                output_mask = output.argmax(1).cpu().data.numpy()

                if source_type[0] == 'zero':
                    incorrect_num = np.sum(output_mask)
                    if incorrect_num == 0:
                        acc = 1
                    else:
                        acc = 0
                    mean_acc.append(acc)
                else:
                    I, U = computeIoU(output_mask, target)
                    if U == 0:
                        this_iou = 0.0
                    else:
                        this_iou = I*1.0/U
                    mean_IoU.append(this_iou)
                    cum_I += I
                    cum_U += U

                    for n_eval_iou in range(len(eval_seg_iou_list)):
                        eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                        seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)

                    seg_total += 1


    mIoU = np.mean(mean_IoU)
    mean_acc = np.mean(mean_acc)
    print('Final results:')
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    results_str += '    mean IoU = %.2f\n' % (mIoU * 100.)
    print(results_str)
    if args.dataset == 'ref-zom':
        print('Mean accuracy for one-to-zero sample is %.2f\n' % (mean_acc*100))

    return mIoU, 100 * cum_I / cum_U


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, print_freq,
                    iterations, bert_model):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0
    total_its = 0

    for data in metric_logger.log_every(data_loader, print_freq, header):
        total_its += 1
        image, target, source_type, sentences, sentences_masked, attentions = data
        source_type = np.array(source_type)
        target_flag = np.where(source_type=='zero', 0, 1)
        target_flag = torch.tensor(target_flag)
        image, target, sentences, sentences_masked,target_flag, attentions = image.cuda(non_blocking=True),\
                                               target.cuda(non_blocking=True),\
                                               sentences.cuda(non_blocking=True),\
                                               sentences_masked.cuda(non_blocking=True),\
                                               target_flag.cuda(non_blocking=True),\
                                               attentions.cuda(non_blocking=True)

        sentences = sentences.squeeze(1)
        sentences_masked = sentences_masked.squeeze(1)
        attentions = attentions.squeeze(1)

        last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]  # (6, 10, 768)
        last_hidden_states1 = bert_model(sentences_masked, attention_mask=attentions)[0]  # (6, 10, 768)
        embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
        embedding1 = last_hidden_states1.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
        attentions = attentions.unsqueeze(dim=-1)  # (batch, N_l, 1)
        loss_contra, loss_lansim, output = model(image, embedding, embedding1, l_mask=attentions,target_flag=target_flag, training_flag=True)

        loss_seg = criterion(output, target)
        loss = loss_seg + loss_lansim * 0.01 + loss_contra * 0.01
        optimizer.zero_grad()  # set_to_none=True is only available in pytorch 1.6+
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        torch.cuda.synchronize()
        train_loss += loss.item()
        iterations += 1
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss_seg=loss_seg.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss_lansim=loss_lansim.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss_contra=loss_contra.item(), lr=optimizer.param_groups[0]["lr"])


        del image, target, sentences, attentions, loss, output, data
        if bert_model is not None:
            del last_hidden_states, embedding

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main(args):
    dataset, num_classes = get_dataset("train",
                                       transform(args=args),
                                       args=args,
                                       eval_mode=False)
    dataset_test, _ = get_dataset(args.split,
                                  get_transform(args=args),
                                  args=args, eval_mode=True)

    # batch sampler
    print(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built train dataset.")
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    if args.dataset == 'ref-zom':
        train_sampler = Refzom_DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                                        shuffle=True)
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                                        shuffle=True)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    # data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers)

    # model initialization
    print(args.model)
    model = segmentation.__dict__[args.model](pretrained=args.pretrained_backbone, args=args)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    single_model = model.module

    model_class = BertModel
    bert_model = model_class.from_pretrained(args.ck_bert)
    bert_model.pooler = None  # a work-around for a bug in Transformers = 3.0.2 that appears for DistributedDataParallel
    bert_model.cuda()
    bert_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(bert_model)
    bert_model = torch.nn.parallel.DistributedDataParallel(bert_model, device_ids=[args.local_rank])
    single_bert_model = bert_model.module


    # resume training
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        single_model.load_state_dict(checkpoint['model'])
        single_bert_model.load_state_dict(checkpoint['bert_model'])

    # parameters to optimize
    backbone_no_decay = list()
    backbone_decay = list()
    for name, m in single_model.backbone.named_parameters():
        if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
            backbone_no_decay.append(m)
        else:
            backbone_decay.append(m)

    params_to_optimize = [
        {'params': backbone_no_decay, 'weight_decay': 0.0},
        {'params': backbone_decay},
        {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
        {"params": [p for p in single_model.contrastive.parameters() if p.requires_grad]},
        # the following are the parameters of bert
        {"params": reduce(operator.concat,
                            [[p for p in single_bert_model.encoder.layer[i].parameters()
                            if p.requires_grad] for i in range(10)])},
    ]
    # optimizer
    optimizer = torch.optim.AdamW(params_to_optimize,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  amsgrad=args.amsgrad
                                  )

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    # housekeeping
    start_time = time.time()
    iterations = 0
    best_oIoU = -0.1

    # resume training (optimizer, lr scheduler, and the epoch)
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resume_epoch = checkpoint['epoch']
    else:
        resume_epoch = -999

    # iou, overallIoU = evaluate(model, data_loader_test, bert_model)
    # training loops
    for epoch in range(max(0, resume_epoch+1), args.epochs):
        data_loader.sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, args.print_freq,
                        iterations, bert_model)
        mean_IoU, overall_IoU = evaluate(model, data_loader_test, bert_model)

        save_checkpoint = (best_oIoU < overall_IoU)
        if save_checkpoint:
            print('Better epoch: {}\n'.format(epoch))
            if single_bert_model is not None:
                dict_to_save = {'model': single_model.state_dict(), 'bert_model': single_bert_model.state_dict(),
                                'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                                'lr_scheduler': lr_scheduler.state_dict()}
            else:
                dict_to_save = {'model': single_model.state_dict(),
                                'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                                'lr_scheduler': lr_scheduler.state_dict()}

            utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
                                                            'model_best_{}.pth'.format(args.model_id)))
            best_oIoU = overall_IoU
            print('The best_performance is {}'.format(best_oIoU))

    # summarize
    print('The final_best_performance is {}'.format(best_oIoU))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    # set up distributed learning
    utils.init_distributed_mode(args)
    print('Image size: {}'.format(str(args.img_size)))
    main(args)