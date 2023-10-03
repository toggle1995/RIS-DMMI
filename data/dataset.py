import os
import torch.utils.data as data
import torch
import numpy as np
from PIL import Image
import pdb
import copy
from random import choice
from bert.tokenization_bert import BertTokenizer
from textblob import TextBlob

from refer.refer import REFER

from args import get_parser

# Dataset configuration initialization
parser = get_parser()
args = parser.parse_args()


class ReferDataset(data.Dataset):

    def __init__(self,
                 args,
                 image_transforms=None,
                 target_transforms=None,
                 split='train',
                 eval_mode=False):

        self.classes = []
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split
        self.refer = REFER(args.refer_data_root, args.dataset, args.splitBy)
        self.dataset_type = args.dataset
        self.max_tokens = 20
        ref_ids = self.refer.getRefIds(split=self.split)
        self.img_ids = self.refer.getImgIds()

        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in self.img_ids)
        self.ref_ids = ref_ids


        self.input_ids = []
        self.input_ids_masked = []
        self.attention_masks = []
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

        self.eval_mode = eval_mode
        
        for r in ref_ids:
            ref = self.refer.Refs[r]

            sentences_for_ref = []
            sentences_for_ref_masked = []
            attentions_for_ref = []

            for i, (el, sent_id) in enumerate(zip(ref['sentences'], ref['sent_ids'])):
                sentence_raw = el['raw']
                attention_mask = [0] * self.max_tokens
                padded_input_ids = [0] * self.max_tokens
                padded_input_ids_masked = [0] * self.max_tokens

                blob = TextBlob(sentence_raw.lower())
                chara_list = blob.tags
                mask_ops = []
                mask_ops1 = []
                for word_i, (word_now, chara) in enumerate(chara_list):
                    if (chara == 'NN' or chara == 'NNS') and word_i < 19 and word_now.lower():
                        mask_ops.append(word_i)
                        mask_ops1.append(word_now)
                mask_ops2 = self.get_adjacent_word(mask_ops)


                input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)

                # truncation of tokens
                input_ids = input_ids[:self.max_tokens]

                padded_input_ids[:len(input_ids)] = input_ids
                attention_mask[:len(input_ids)] = [1]*len(input_ids)
                if len(mask_ops) == 0:
                    attention_remask = attention_mask
                    input_ids_masked = input_ids
                else:
                    could_mask = choice(mask_ops2)
                    input_ids_masked = copy.deepcopy(input_ids)
                    for i in could_mask:
                        input_ids_masked[i + 1] = 0
                padded_input_ids_masked[:len(input_ids_masked)] = input_ids_masked


                sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
                sentences_for_ref_masked.append(torch.tensor(padded_input_ids_masked).unsqueeze(0))
                attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))

            self.input_ids.append(sentences_for_ref)
            self.input_ids_masked.append(sentences_for_ref_masked)
            self.attention_masks.append(attentions_for_ref)


    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.ref_ids)
    
    def get_adjacent_word(self, mask_list):
        output_mask_list = []
        length = len(mask_list)
        i = 0
        while i < length:
            begin_pos = i
            while i+1 < length and mask_list[i+1] == mask_list[i] + 1:
                i += 1
            end_pos = i+1
            output_mask_list.append(mask_list[begin_pos:end_pos])
            i = end_pos

        return output_mask_list

    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]

        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]

        img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])).convert("RGB")

        ref = self.refer.loadRefs(this_ref_id)
        if self.dataset_type == 'ref_zom':
            source_type = ref[0]['source']
        else:
            source_type = 'one'

        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])

        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1
        annot = Image.fromarray(annot.astype(np.uint8), mode="P")


        if self.image_transforms is not None:

            if self.split == 'train':
                img, target = self.image_transforms(img, annot)
            elif self.split == 'val':
                img, target = self.image_transforms(img, annot)
            else:
                img, target = self.image_transforms(img, annot)

        if self.eval_mode:
            embedding = []
            embedding_masked = []
            att = []
            for s in range(len(self.input_ids[index])):
                e = self.input_ids[index][s]
                a = self.attention_masks[index][s]
                embedding.append(e.unsqueeze(-1))
                embedding_masked.append(e.unsqueeze(-1))
                att.append(a.unsqueeze(-1))
            
            tensor_embeddings = torch.cat(embedding, dim=-1)
            tensor_embeddings_masked = torch.cat(embedding_masked, dim=-1)
            attention_mask = torch.cat(att, dim=-1)
        else:
            choice_sent = np.random.choice(len(self.input_ids[index]))
            tensor_embeddings = self.input_ids[index][choice_sent]
            tensor_embeddings_masked = self.input_ids_masked[index][choice_sent]
            attention_mask = self.attention_masks[index][choice_sent]

        return img, target, source_type, tensor_embeddings, tensor_embeddings_masked, attention_mask