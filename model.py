import numpy as np
import torch
import torch.nn as nn
import logging

from torchfly.modules.transformers import GPT2SimpleLM, UnifiedGPT2SmallConfig, UnifiedGPT2LargeConfig
from torchfly.utils import init_logging
# from torchfly.criterions import SequenceCrossEntropyLoss
from torchfly.modules.losses import SequenceCrossEntropyLoss
from torchfly.utils import get_pretrained_states
import utils

init_logging()
logger = logging.getLogger(__name__)


def sequence_ce_lm_loss(
    logits: torch.FloatTensor,
    lm_logits: torch.FloatTensor,
    targets: torch.LongTensor,
    mask: torch.FloatTensor,
    kl_coef: float,
):
    """
    Sequence Cross Entropy with Language Model KL
    """

    # shape : (batch, sequence_length, num_classes)
    log_probs = torch.log_softmax(logits, dim=-1)
    lm_probs = torch.softmax(lm_logits, dim=-1)

    # shape : (batch, sequence_length)
    negative_log_likelihood = -torch.gather(
        log_probs, dim=2, index=targets.unsqueeze(2)
    ).squeeze(2)

    # ignored mask and normalized by length
    lm_kl = (
        torch.kl_div(input=log_probs, target=lm_probs, reduction=2) /
        log_probs.shape[1]
    )

    # shape : (batch, sequence_length)
    loss = negative_log_likelihood * mask + kl_coef * lm_kl

    loss = loss.sum(1) / (mask.sum(1) + 1e-5)
    loss = loss.mean()

    return loss, lm_kl


class ARDM(nn.Module):
    def __init__(self, args):
        super(ARDM, self).__init__()
        self.args = args

        # define the two language models
        self.model_A = GPT2SimpleLM(UnifiedGPT2SmallConfig)
        self.model_B = GPT2SimpleLM(UnifiedGPT2SmallConfig)
        # load weights
        self.model_A.load_state_dict(get_pretrained_states("unified-gpt2-small"))
        self.model_B.load_state_dict(get_pretrained_states("unified-gpt2-small"))
        
        # language model KL
        if args.kl_model_type == "small":
            self.language_model = GPT2SimpleLM(UnifiedGPT2SmallConfig)
            self.language_model.load_state_dict(
                get_pretrained_states("unified-gpt2-small")
            )
        elif args.kl_model_type == "large":
            self.language_model = GPT2SimpleLM(UnifiedGPT2LargeConfig)
            self.language_model.load_state_dict(
                get_pretrained_states("unified-gpt2-large-fp16"), strict=False
            )
        elif args.kl_model_type == "no":
            self.language_model = GPT2SimpleLM(UnifiedGPT2SmallConfig)
            self.language_model.load_state_dict(
                get_pretrained_states("unified-gpt2-small")
            )
        else:
            raise ValueError("kl model name error")
        # freeze weights
        utils.freeze_model(self.language_model)

        self.criterion = sequence_ce_lm_loss
        self.lm_coef = 0.1
        self.lm_coef_decay = 0.9999

        if args.use_discount:
            self.discount_factor = 0.95
        else:
            self.discount_factor = 1.00
        self.lm_stream = torch.cuda.Stream()

    def forward(self, dialog):
        raise NotImplementedError

    def get_loss(self, inputs, logits, lm_logits):
        logits = logits[:, :-1].contiguous()
        lm_logits = lm_logits[:, :-1].contiguous()
        target = inputs[:, 1:].contiguous()
        target_mask = torch.ones_like(target).float()
        return self.criterion(
            logits, lm_logits, target, target_mask, self.lm_coef
        )

    def train_one_step(self, batch):
        self.lm_coef *= self.lm_coef_decay
        self.model_A.train()
        self.model_B.train()

        dialog = batch["input_ids"]
        dialog_position_ids = batch["position_ids"]

        past = None
        lm_past = None
        total_loss = 0.0
        total_kl = 0.0
        discount_coef = (self.discount_factor**np.arange(len(dialog)))[::-1]

        for turn_idx in range(len(dialog)):
            # language model KL Divergence
            torch.cuda.synchronize()
            with torch.cuda.stream(self.lm_stream):
                lm_logits, lm_past = self.language_model(
                    dialog[turn_idx], position_ids=dialog_position_ids[turn_idx], past=lm_past
                )

            if turn_idx % 2 == 0:
                logits, past = self.model_A(dialog[turn_idx], position_ids=dialog_position_ids[turn_idx], past=past)

            else:
                logits, past = self.model_B(dialog[turn_idx], position_ids=dialog_position_ids[turn_idx], past=past)

            if self.args.kl_model_type == "no":
                loss, lm_kl = self.get_loss(dialog[turn_idx], logits, logits)
            else:
                loss, lm_kl = self.get_loss(dialog[turn_idx], logits, lm_logits)
            total_loss += discount_coef[turn_idx] * loss
            total_kl += lm_kl.item()

        # normalize by turns
        total_loss /= len(dialog)
        total_kl /= len(dialog)

        return total_loss, total_kl

    def switch_roles(self):
        temp_var = self.model_A
        self.model_A = self.model_B
        self.model_B = temp_var


def dialog_to_tensor(tokenzier, dialog, device=None):
    res = [torch.LongTensor([tokenizer.encode(item)]) for item in dialog]
    if device:
        res = [item.to(device) for item in res]
    return res
