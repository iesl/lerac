import os
import csv
import json
from collections import defaultdict
from tqdm import tqdm

from transformers.tokenization_bert import BertTokenizer

from IPython import embed


#DATA_DIR = '/mnt/nfs/scratch1/rangell/BLINK/data/'

DATASET = 'BC5CDR'
REPO_ROOT = '/mnt/nfs/scratch1/rangell/lerac/coref_entity_linking/'
#PMIDS_FILE = REPO_ROOT + 'data/raw_BC5CDR/BC5CDR/BC5CDR_traindev_PMIDs.txt'
#PMIDS_FILE = REPO_ROOT + 'data/raw_BC5CDR/BC5CDR/BC5CDR_sample_PMIDs.txt'
PMIDS_FILE = REPO_ROOT + 'data/raw_BC5CDR/BC5CDR_TEST/CDR_TestSet.pmids.txt'
#PUBTATOR_FILE = REPO_ROOT + 'data/raw_BC5CDR/BC5CDR/CDR.2.PubTator'
PUBTATOR_FILE = REPO_ROOT + 'data/raw_BC5CDR/BC5CDR_TEST/CDR_TestSet.PubTator.joint.txt'
MATCHES_FILE = REPO_ROOT + 'data/raw_BC5CDR/mention_matches_bc5cdr.txt'
ENTITY_FILES = [
    ('Chemical', REPO_ROOT + 'data/raw_BC5CDR/BC5CDR/CTD_chemicals-2015-07-22.tsv'),
    ('Disease', REPO_ROOT + 'data/raw_BC5CDR/BC5CDR/CTD_diseases-2015-06-04.tsv')
]

OUTPUT_DIR = '/mnt/nfs/scratch1/rangell/lerac/data/{}'.format(DATASET)


if __name__ == '__main__':

    # get tokenizer
    tokenizer = BertTokenizer(
        '/mnt/nfs/scratch1/rangell/lerac/coref_entity_linking/'\
            'models/biobert_v1.1_pubmed/vocab.txt',
        do_lower_case=False
    )

    # get all pmids
    with open(PMIDS_FILE, 'r') as f:
        pmids = set(map(lambda x : x.strip(), f.readlines()))

    # get all of the documents
    raw_docs = defaultdict(str)
    gold_mention_labels = {}
    with open(PUBTATOR_FILE, 'r') as f:
        for line in f:
            line_split = line.strip().split('\t')
            if len(line_split) == 6:
                if line_split[0] not in pmids:
                    continue
                gold_key = (line_split[0],line_split[1], line_split[2])
                gold_mention_labels[gold_key] = [
                    s.replace('UMLS:', '') for s in line_split[-1].split('|')
                ]
            else:
                line_split = line.split('|')
                if len(line_split) == 3:
                    assert line_split[1] in ['a', 't']
                    if line_split[0] not in pmids:
                        continue
                    _text_to_add = ' ' if line_split[1] == 'a' else ''
                    _text_to_add += line_split[2].strip()
                    raw_docs[line_split[0]] += _text_to_add

    # tokenize all of the documents
    tokenized_docs = {}
    for pmid, raw_text in raw_docs.items():
        wp_tokens = tokenizer.tokenize(raw_text)
        tokenized_text = ' '.join(wp_tokens).replace(' ##', '')
        tokenized_docs[pmid] = tokenized_text

    # get all of the mentions and their tfidf candidates in raw form
    print('Reading mentions, tfidf candidates, and building entity set...')
    mention_cands = defaultdict(list)
    cuid2names = defaultdict(list)
    cuid2type = {}
    with open(MATCHES_FILE, 'r') as f:
        reader = csv.reader(f, delimiter="\t", quotechar='"')
        keys = next(reader)
        for row in tqdm(reader):
            cuid2names[row[8]].append(row[11])
            cuid2type[row[8]] = row[7]
            if row[0] not in pmids:
                continue
            mention_key = (row[0], row[1], row[2])
            mention_cand_val = {k : v for k, v in zip(keys, row)}
            mention_cands[mention_key].append(mention_cand_val)
    print('Done.')

    # build entity dict
    entity_dict = {}
    for cuid, names in cuid2names.items():
        name_counts = defaultdict(int)
        for name in names:
            name_counts[name] += 1
        names_w_counts = sorted(
            name_counts.items(), reverse=True, key=lambda x: x[1]
        )
        uniq_names, _ = zip(*names_w_counts)
        
        if len(uniq_names) > 1:
            entity_obj = {
                'document_id': cuid,
                'title': uniq_names[0],
                'text': '{} ( {} )'.format(
                        uniq_names[0], ' ; '.join(uniq_names[1:])
                    ),
                'type': cuid2type[cuid]
            }
        else:
            entity_obj = {
                'document_id': cuid,
                'title': uniq_names[0],
                'text': uniq_names[0],
                'type': cuid2type[cuid]
            }
        entity_dict[cuid] = entity_obj

    # process entity files
    for entity_type, raw_entity_file in ENTITY_FILES:
        with open(raw_entity_file, 'r') as f:
            for line in f:
                if line[0] == '#':
                    continue
                line_cols = line.strip().split('\t')
                name = line_cols[0]
                cuid = line_cols[1].replace('MESH:', '').replace('OMIM:', '')
                if cuid in entity_dict.keys():
                    continue
                synonyms = line_cols[-1].split('|')
                entity_obj = {
                    'document_id': cuid,
                    'title': name,
                    'text': '{} ( {} )'.format(
                            name, ' ; '.join(synonyms)
                        ),
                    'type': entity_type
                }
                entity_dict[cuid] = entity_obj

    # organize mentions by pmid
    mentions = defaultdict(list)
    for key, value in mention_cands.items():
        mentions[key[0]].append(value[0])
    
    # sort the mentions in each doc
    for key in list(mentions.keys()):
        mentions[key] = sorted(mentions[key], key=lambda x : int(x['char_start']))
        _mentions = []
        for m in mentions[key]:
            condition = any(
                [((int(m['char_start']) >= int(x['char_start'])
                     and int(m['char_start']) <= int(x['char_end']))
                    or (int(m['char_end']) >= int(x['char_start'])
                     and int(m['char_end']) <= int(x['char_end'])))
                 and (m['char_start'] != x['char_start'] or m['char_end'] != x['char_end'])
                 and (int(m['char_end']) - int(m['char_start']) <= int(x['char_end']) - int(x['char_start']))
                    for x in mentions[key]]
            )
            if not condition:
                _mentions.append(m)
        mentions[key] = _mentions

    # do a token match and get the start offset
    def get_offset(index_list, query_list, start_offset):
        for i in range(start_offset, len(index_list)):
            match = True
            for j, query_elt in enumerate(query_list):
                if query_elt != index_list[i+j]:
                    match = False
                    break
            if match:
                return i
        return -1


    # process mentions
    restricted_mention_type = 'Disease'
    mention_objs = []
    tfidf_candidate_objs = []
    for pmid, _mentions in tqdm(mentions.items(), desc='Process mentions'):
        start_offset = 0
        for i, m in enumerate(_mentions):

            # tokenize meniton and expanded mention
            tokenized_mention = tokenizer.tokenize(m['mention'])
            tokenized_mention = ' '.join(tokenized_mention).replace(' ##', '')
            tokenized_mention = tokenized_mention.split()
            tokenized_mention_exp = tokenizer.tokenize(m['mention_exp'])
            tokenized_mention_exp = ' '.join(tokenized_mention_exp).replace(' ##', '')
            tokenized_mention_exp = tokenized_mention_exp.split()

            # do find and replace
            tokenized_doc = tokenized_docs[pmid].split()
            start_index = get_offset(
                tokenized_doc, tokenized_mention, start_offset
            )
            if start_index == -1: # somehow the mention was not found, ignore
                continue
            tokenized_doc = tokenized_doc[:start_index] \
                            + tokenized_mention_exp \
                            + tokenized_doc[start_index+len(tokenized_mention):] 
            end_index = start_index + len(tokenized_mention_exp) - 1
            start_offset = end_index + 1
            assert ' '.join(tokenized_doc[start_index:end_index+1]) == ' '.join(tokenized_mention_exp)

            # create mention object and add to list of mentions
            start_char = m['char_start']
            end_char = m['char_end']
            mention_obj = {
                'mention_id' : '.'.join([pmid, str(i)]),
                'context_document_id' : pmid,
                'start_index' : start_index,
                'end_index' : end_index,
                'text' : ' '.join(tokenized_mention_exp),
                'category': m['mention_tid'],
                'label_document_id': gold_mention_labels.get(
                        (pmid, start_char, end_char), []
                    )
            }

            if mention_obj['category'] != restricted_mention_type:
                continue

            # add the mention object to the list
            mention_objs.append(mention_obj)

            # get candidates
            tfidf_cand_cuids = []
            for cand in mention_cands[(pmid, start_char, end_char)]:
                if cand['match_tid'] == restricted_mention_type:
                    tfidf_cand_cuids.append(cand['match_cuid'])

            # create tfidf candidates object and add to list
            tfidf_cands_obj = {
                'mention_id' : '.'.join([pmid, str(i)]),
                'tfidf_candidates' : tfidf_cand_cuids
            }
            tfidf_candidate_objs.append(tfidf_cands_obj)

            tokenized_docs[pmid] = ' '.join(tokenized_doc)

    # write to output files
    with open('mentions.jsonl', 'w') as f:
        for m in mention_objs:
            f.write(json.dumps(m) + '\n')

    with open('tfidf_candidates.jsonl', 'w') as f:
        for c in tfidf_candidate_objs:
            f.write(json.dumps(c) + '\n')

    with open('documents.jsonl', 'w') as f:
        for pmid, doc_text in tokenized_docs.items():
            doc_dict = {
                'document_id' : pmid,
                'title' : pmid,
                'text' : doc_text
            }
            f.write(json.dumps(doc_dict) + '\n')

    with open('entity_documents.jsonl', 'w') as f:
        for ent in entity_dict.values():
            f.write(json.dumps(ent) + '\n')
