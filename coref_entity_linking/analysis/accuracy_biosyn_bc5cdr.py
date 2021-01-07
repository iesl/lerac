from collections import defaultdict
import json

from IPython import embed


PUBTATOR_FILE = '/mnt/nfs/scratch1/rangell/lerac/coref_entity_linking/data/raw_BC5CDR/BC5CDR_TEST/CDR_TestSet.PubTator.joint.txt'
PREDS_FILE = '/mnt/nfs/scratch1/rangell/BioSyn/tmp/biosyn-bc5cdr/predictions_eval.json'


def get_gold_mapping():
    gold_mention_cuis = {}
    with open(PUBTATOR_FILE, 'r') as f:
        for line in f:
            split_line = line.strip().split('\t')
            if len(split_line) == 6:
                gold_mention_cuis[tuple(split_line[0:3])] = split_line[-1].split('|')


if __name__ == '__main__':

    gold_mention_cuis = get_gold_mapping()

    with open(PREDS_FILE, 'r') as f:
        preds_json = json.load(f)


    embed()
    exit()
