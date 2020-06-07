import numpy as np
import torch
from tqdm import tqdm

from utils.comm import get_rank, all_gather, synchronize
from utils.misc import flatten

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
        self.model.eval()

        local_step = 0
        push_to_cpu_steps = 32
        idxs_list = []
        embeds_list = []
        master_idxs_list = []
        master_embeds_list = []

        def _synchronize_lists(_embeds_list, _idxs_list):
            gathered_data = all_gather({
                    'embeds_list': _embeds_list,
                    'idxs_list': _idxs_list,
                })
            if get_rank() == 0:
                _embeds_list = [d['embeds_list'] for d in gathered_data]
                _embeds_list = flatten(_embeds_list)
                _embeds_list = [x.cpu() for x in _embeds_list]
                _idxs_list = [d['idxs_list'] for d in gathered_data]
                _idxs_list = flatten(_idxs_list)
                _idxs_list = [x.cpu() for x in _idxs_list]
                master_embeds_list.extend(_embeds_list)
                master_idxs_list.extend(_idxs_list)
            synchronize()
            return [], []
            
        batch_iterator = tqdm(dataloader,
                              desc='Getting embeddings...',
                              disable=(not evaluate or args.disable_logging))
        for batch in batch_iterator:
            batch = tuple(t.to(args.device, non_blocking=True) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids_a':      batch[1],
                          'attention_mask_a': batch[2],
                          'token_type_ids_a': batch[3]}
                embeds_list.append(self.model(**inputs))
                idxs_list.append(batch[0])
                local_step += 1
                if local_step % push_to_cpu_steps == 0:
                    embeds_list, idxs_list = _synchronize_lists(
                            embeds_list, idxs_list)

        embeds_list, idxs_list = _synchronize_lists(
                embeds_list, idxs_list)

        idxs, embeds = None, None
        if get_rank() == 0:
            idxs = torch.cat(master_idxs_list, dim=0).numpy()
            idxs, indices = np.unique(idxs, return_index=True)
            embeds = torch.cat(master_embeds_list, dim=0).numpy()
            embeds = embeds[indices]
        synchronize()
        return idxs, embeds

    def compute_scores_for_inference(self, dataset):
        pass

    def train_on_subset(self, dataset):
        pass

    def compute_topk_score_evaluation(self, dataset):
        pass

