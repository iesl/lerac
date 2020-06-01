import argparse
from collections import defaultdict
import os
import pickle
import re
from scipy import sparse
from sklearn.metrics.pairwise import pairwise_distances
import string
from tqdm import tqdm

from IPython import embed

from data import get_mentions_and_entities, _read_candidates, get_document2mentions
import genia_tokenizer


DOMAINS = ["val","T005","T007","T017","T022","T031","T033","T037","T038","T058","T062","T074","T082","T091","T092","T097","T098","T103","T168","T170","T201","T204"]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", 
                        default="data/mm_st21pv_long_entities", 
                        type=str, help="data directory")
    parser.add_argument('-u', '--char_vec_binary',
                        default='bin/char_vectorizer.pkl',
                        help='serialized character tfidf vectorizer')
    parser.add_argument('-v', '--word_vec_binary',
                        default='bin/word_vectorizer.pkl',
                        help='serialized word tfidf vectorizer')
    args = parser.parse_args()
    return args


def normalize_string(s):
    s = ' '.join(genia_tokenizer.tokenize(s))
    s = re.sub('[' + string.punctuation + ']', '', s.lower())
    return s


def get_mentions_and_candidates(args):
    split = 'val'
    outputs = get_mentions_and_entities(args.data_dir, split, DOMAINS)
    mentions, _, entities = outputs
    mentions_dict = {m['mention_id']: m for m in mentions}

    candidate_file = os.path.join(args.data_dir, 'tfidf_candidates', split + '.json')
    candidates = _read_candidates(candidate_file)

    document2mentions = get_document2mentions(mentions)
    document2candidates = defaultdict(list)
    for doc_id, doc_mentions in document2mentions.items():
        for m in doc_mentions:
            muid = m['mention_id'] 
            if muid in candidates.keys():
                document2candidates[doc_id].extend(candidates[muid])
        for m in doc_mentions:
            muid = m['mention_id']
            candidates[muid] = document2candidates[doc_id]

    return document2mentions, entities, document2candidates


def score_mentions_and_candidates(args, char_vectorizer, word_vectorizer):
    outputs = get_mentions_and_candidates(args)
    document2mentions, entities, document2candidates = outputs

    scores_dict = defaultdict(dict)
    for doc_id, doc_mentions in tqdm(document2mentions.items()):
        #doc_candidates = [entities[cuid] for cuid in document2candidates[doc_id]]
        _mentions = [(m['mention_id'], normalize_string(m['text']))
                                for m in doc_mentions]
        #_candidates = [(c['document_id'], normalize_string(c['text']))
        #                        for c in doc_candidates]
        mention_ids, mention_strings = zip(*_mentions)
        #candidate_ids, candidate_strings = zip(*_candidates)

        mention_vectors = sparse.hstack(
                (char_vectorizer.transform(mention_strings),
                 word_vectorizer.transform(mention_strings))
        ).tocsr()

        #candidate_vectors = sparse.hstack(
        #        (char_vectorizer.transform(candidate_strings),
        #         word_vectorizer.transform(candidate_strings))
        #).tocsr()
                            
        #pair_dists = pairwise_distances(mention_vectors,
        #                                candidate_vectors,
        #                                metric='cosine',
        #                                n_jobs=10)

        #scores_dict[doc_id] = {
        #        'mention_ids' : mention_ids,
        #        'candidate_ids' : candidate_ids,
        #        'pairwise_dists' : pair_dists
        #}
        pair_dists = pairwise_distances(mention_vectors,
                                        mention_vectors,
                                        metric='cosine',
                                        n_jobs=10)

        scores_dict[doc_id] = {
                'mention_ids' : mention_ids,
                'pairwise_dists' : pair_dists
        }

    return scores_dict


def main(args):
    print('Loading char vectorizer')
    with open(args.char_vec_binary, 'rb') as f:
        char_vectorizer = pickle.load(f)

    print('Loading word vectorizer')
    with open(args.word_vec_binary, 'rb') as f:
        word_vectorizer = pickle.load(f)

    scores_dict = score_mentions_and_candidates(
            args, char_vectorizer, word_vectorizer
    )

    #with open('mention_candidate_scores.val.pkl', 'wb') as f:
    #    pickle.dump(scores_dict, f, pickle.HIGHEST_PROTOCOL)

    with open('mention_mention_scores.val.pkl', 'wb') as f:
        pickle.dump(scores_dict, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    args = get_args()
    main(args)
