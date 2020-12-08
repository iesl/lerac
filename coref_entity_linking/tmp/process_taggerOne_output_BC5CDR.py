import csv
import json
from collections import defaultdict

from IPython import embed


TARGET_FILE = 'mention_level_preds_BC5CDR.target.txt'
PRED_FILE = 'mention_level_preds_BC5CDR.predicted.txt'
TRAIN_JSON_FILE = '../data/BC5CDR/mentions/train.json'
TEST_JSON_FILE = '../data/BC5CDR/mentions/test.json'


target_cuids = defaultdict(list)
with open(TARGET_FILE, 'r') as f:
    tsv_reader = csv.reader(f, delimiter='\t')
    for row in tsv_reader:
        target_cuids[(row[2], row[3], row[4])].append(
            row[5].replace('MESH:', '').replace('OMIM:', '')
        )


# compute singleton mention targets
doc_entity_mentions = defaultdict(lambda : defaultdict(list))
for mention_key, tgt_cuid in target_cuids.items():
    for cuid in tgt_cuid:
        doc_entity_mentions[mention_key[0]][cuid].append(mention_key)

singleton_targets = []
for entity2mentions in doc_entity_mentions.values():
    for mention_keys in entity2mentions.values():
        if len(mention_keys) == 1:
            singleton_targets.extend(mention_keys)
singleton_targets = set(singleton_targets)


# load taggerOne predictions
pred_cuids = defaultdict(list)
with open(PRED_FILE, 'r') as f:
    tsv_reader = csv.reader(f, delimiter='\t')
    for row in tsv_reader:
        pred_cuids[(row[2], row[3], row[4])].append(
            row[5].replace('MESH:', '').replace('OMIM:', '')
        )


# get seen and unseen cuids
seen_cuids = []
with open(TRAIN_JSON_FILE, 'r') as f:
    for line in f:
        mention_obj = json.loads(line)
        seen_cuids.extend(mention_obj['label_document_id'])
seen_cuids = set(seen_cuids)


# get singleton counts (as evaluated by lerac)
doc_entity_mentions = defaultdict(lambda : defaultdict(list))
with open(TEST_JSON_FILE, 'r') as f:
    for line in f:
        mention_obj = json.loads(line)
        ctxt_doc = mention_obj['context_document_id']
        cuids = mention_obj['label_document_id']
        for cuid in cuids:
            doc_entity_mentions[ctxt_doc][cuid].append(1)

total_singletons = 0
seen_singletons = 0
for entity2mentions in doc_entity_mentions.values():
    for cuid, mention_keys in entity2mentions.items():
        if len(mention_keys) == 1:
            total_singletons += 1
            if cuid in seen_cuids:
                seen_singletons += 1


hits, total = 0, 0
seen_hits, seen_total = 0, 0
unseen_hits, unseen_total = 0, 0
for mention_key, tgt_cuid in target_cuids.items():

    if mention_key in singleton_targets:
        continue

    _hit = False
    if any([x in pred_cuids.get(mention_key, -1) for x in tgt_cuid]):
        hits += 1
        _hit = True
        
    if any([x in seen_cuids for x in tgt_cuid]):
        if _hit:
            seen_hits += 1
        seen_total += 1
    else:
        if _hit:
            unseen_hits += 1
        unseen_total += 1

    total += 1
    
total = 9758 # the actual number of ground truth spans

embed()
exit()
