import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Contrastive_Loss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    """
    def __init__(self, temperature=0.05):
        super(Contrastive_Loss, self).__init__()
        self.temperature =  temperature
        self.adpool = nn.AdaptiveAvgPool2d((1, 1))
        self.align_lan = nn.Sequential(
            nn.Conv1d(768, 768, kernel_size=1, stride=1),
        )


    def forward(self, vis_feature, lan_feature, target_flag):
        """
        """
        vis_feature1 = F.normalize(vis_feature, dim=1)
        lan_feature1 = self.align_lan(lan_feature)
        lan_feature1 = self.adpool(lan_feature1.unsqueeze(3)).view(lan_feature.shape[0], lan_feature.shape[1])
        lan_feature1 = F.normalize(lan_feature1, dim=1)

        img_text_logits = torch.matmul(vis_feature1, lan_feature1.permute(1, 0)) / self.temperature
        text_img_logits = img_text_logits.permute(1, 0)
        labels = torch.arange(0, len(lan_feature)).cuda()
        loss_a = nn.functional.cross_entropy(img_text_logits, labels, reduce=False)
        loss_b = nn.functional.cross_entropy(text_img_logits, labels, reduce=False)
        loss_a = torch.mean(loss_a * target_flag)
        loss_b = torch.mean(loss_b * target_flag)
            
        loss_con = loss_a + loss_b
        return loss_con

class Cosine_Sim_Loss(nn.Module):
    """cosine similarity function.
    """

    def __init__(self):
        super(Cosine_Sim_Loss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, lan1, lan2, mask_full, target_flag):
        """
        """
        maskf1 = mask_full.permute(0, 2, 1)
        target_flag = target_flag.view(target_flag.shape[0], 1)

        lan1_1 = lan1 * maskf1
        lan2_1 = lan2 * maskf1
        lan1_1_clone = lan1_1.detach()
        score = self.cos(lan1_1_clone, lan2_1)
        score = score * target_flag
        score1 = torch.sum(score, dim=-1)
        length = torch.sum(maskf1, dim=-1).squeeze(-1)
        mean_score = score1 / length
        # pdb.set_trace()

        loss_cossim = 1 - torch.mean(mean_score)
        return loss_cossim