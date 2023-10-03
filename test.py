import torch
import torch.utils.data

from bert.modeling_bert import BertModel
from data.dataset_zom import Referzom_Dataset
from data.dataset import ReferDataset

from lib import segmentation
import transforms as T
import utils

import numpy as np
import torch.nn.functional as F
import pdb


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


def evaluate(model, data_loader, bert_model, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    mean_acc = []

    header = 'Test:'

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            image, target, source_type,  sentences, sentences_masked, attentions = data

            image, target, sentences, attentions = image.to(device), target.to(device), \
                                                   sentences.to(device), attentions.to(device)
            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            target = target.cpu().data.numpy()
            for j in range(sentences.size(-1)):

                last_hidden_states = bert_model(sentences[:, :, j], attention_mask=attentions[:, :, j])[0]
                embedding = last_hidden_states.permute(0, 2, 1)
                output = model(image, embedding, embedding, l_mask=attentions[:, :, j].unsqueeze(-1))[2]

                output = output.cpu()
                output_mask = output.argmax(1).data.numpy()
                
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

            del image, target, sentences, attentions, output, output_mask
            if bert_model is not None:
                del last_hidden_states, embedding

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)


    mean_acc = np.array(mean_acc)
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

def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)


def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U


def main(args):
    device = torch.device(args.device)
    dataset_test, _ = get_dataset(args.split, get_transform(args=args), args, eval_mode=True)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   sampler=test_sampler, num_workers=args.workers)
    print(args.model)
    single_model = segmentation.__dict__[args.model](pretrained='',args=args)
    checkpoint = torch.load(args.test_parameter, map_location='cpu')
    single_model.load_state_dict(checkpoint['model'])
    model = single_model.to(device)

    model_class = BertModel
    single_bert_model = model_class.from_pretrained(args.ck_bert)
    # work-around for a transformers bug; need to update to a newer version of transformers to remove these two lines
    if args.ddp_trained_weights:
        single_bert_model.pooler = None
    single_bert_model.load_state_dict(checkpoint['bert_model'])
    bert_model = single_bert_model.to(device)


    evaluate(model, data_loader_test, bert_model, device=device)


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    print('Image size: {}'.format(str(args.img_size)))
    main(args)