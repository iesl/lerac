import numpy as np
import torch
from tqdm import tqdm

from utils.comm import get_rank, all_gather

from IPython import embed


class EmbeddingSubTrainer(object):
    """
    Class to help with training and evaluation processes.
    """
    def __init__(self, args, model):
        # we assume that `model` is a DDP pytorch model and is on the GPU
        self.args = args
        self.model = model

    def get_embeddings(self, dataloader, evaluate=True):
        args = self.args
        idxs_list = []
        batch_iterator = tqdm(dataloader,
                              desc='Getting embeddings...',
                              disable=(not evaluate or args.disable_logging))
        for batch in batch_iterator:
            batch = tuple(t.to(args.device, non_blocking=True) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids_a':      batch[1],
                          'attention_mask_a': batch[2],
                          'token_type_ids_a': batch[3]}
                outputs = self.model(**inputs)
                idxs_list.append(batch[0].cpu().numpy())
                embed()
                exit()

    def compute_scores_for_inference(self, dataset):
        pass

    def train_on_subset(self, dataset):
        pass

    def compute_topk_score_evaluation(self, dataset):
        pass

