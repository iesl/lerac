'''
Use python3
'''
import time
import sys
import os
import string
import re
import glob
import csv
import argparse
import pickle
import copy
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances
import pysparnn.cluster_index as ci
import gzip
from nltk.stem import PorterStemmer
import genia_tokenizer
from scipy import sparse
import numpy as np
import fasttext
import tempfile
from joblib import Parallel, delayed, load, dump
from tqdm import tqdm, trange
from IPython import embed

stemmer = PorterStemmer()

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--num_shards', default=5, type=int,
                        help='reduce memory by sharding mention-entity scoring')
    parser.add_argument('-n', '--num_candidates', default=100, type=int,
                        help='num candidates per mention to export')
    parser.add_argument('-g', '--char_grams', default='2,5',
                        help='comma seperated min and max range for character ngram features')
    parser.add_argument('-w', '--word_grams', default='1,2',
                        help='comma seperated min and max range for word ngram features')
    parser.add_argument('-f', '--max_features', type=int, default=100000,
                        help='max number of char and word features for tfidf')

    parser.add_argument('-a', '--abbreviation_file', default='../data/abbreviations.tsv',
                        help='tsv of resolved abbreviations [docid, abbrev, fullname]')
    parser.add_argument('-c', '--concept_file',
                        default='../data/umls/umls_2017_installation/2017AA/META/MRCONSO.RRF',
                        help='file containing mapping from UMLS concepts to synonyms')
    parser.add_argument('-t', '--type_file',
                        default='../data/umls/umls_2017_installation/2017AA/META/MRSTY.RRF',
                        help='file containing mapping from UMLS concepts to types')
    parser.add_argument('-r', '--type_hierarchy_file',
                        default='../data/umls/umls_2017_installation/2017AA/NET/SRSTRE1',
                        help='file containing hierarchy of UMLS types')
    parser.add_argument('-b', '--name2concept_binary',
                        default='../bin/untyped_name2concept.normalized.pkl',
                        help='serialized concept data structure')
    parser.add_argument('-d', '--concept2name_binary',
                        default='../bin/concept2name.pkl',
                        help='serialized concept data structure')
    parser.add_argument('-u', '--char_vec_binary',
                        default='../bin/char_vectorizer.pkl',
                        help='serialized character tfidf vectorizer')
    parser.add_argument('-v', '--word_vec_binary',
                        default='../bin/word_vectorizer.pkl',
                        help='serialized word tfidf vectorizer')
    parser.add_argument('-o', '--out_file_prefix',
                        default='../results/tfidf_hits_and_misses/',
                        help='directory for alias table results')
    parser.add_argument('-k', '--skipgram_lm_file',
                        default='../bin/skipgram.normalized.bin',
                        help='serialized skipgram language model')
    
    parser.add_argument('-x', '--train_file',
                        default='../data/MedMentions/st21pv/MedMentions_ST21pv.pubtator.train.txt.gz',
                        help='pubtator formatted train file')
    parser.add_argument('-y', '--dev_file',
                        default='../data/MedMentions/st21pv/MedMentions_ST21pv.pubtator.dev.txt.gz',
                        help='pubtator formatted dev file')
    parser.add_argument('-z', '--test_file',
                        default='../data/MedMentions/st21pv/MedMentions_ST21pv.pubtator.test.txt.gz',
                        help='pubtator formatted test file')
    return parser


st21pv_types = ['T005', 'T007', 'T017', 'T022', 'T031', 'T033', 'T037', 'T038',
               'T058', 'T062', 'T074', 'T082', 'T091', 'T092', 'T097', 'T098',
               'T103', 'T168', 'T170', 'T201', 'T204']

def normalize_string(s):
    '''
    normalize entity names with [genia tokenize, lowercase, stem, punctuation removal]
    :param s: entity string name
    :return: normalized string
    '''
    # s = s.decode('utf8')
    s = ' '.join(genia_tokenizer.tokenize(s))
    s = re.sub('[' + string.punctuation + ']', '', s.lower())
    #tokens = [stemmer.stem(t) for t in s.split()]
    #norm_s = ' '.join(tokens) if tokens else s
    return s


def read_abbrev_map(args):
    print('Reading in abbrevation -> full name map')
    _abbrev_map = {}
    with open(args.abbreviation_file, 'r') as inf:
        for l in inf:
            pmid, abbrev, full_name = l.strip().split('\t')
            _abbrev_map['%s\t%s' % (pmid, abbrev)] = full_name
    return _abbrev_map


def read_type_map(args):
    print('Reading in types -> full concept to type map')
    _concept2type = defaultdict(list)
    _type2name = {}
    with open(args.type_file, 'r') as f:
        reader = csv.reader(f, delimiter='|')
        for l in reader:
            _concept2type[l[0]].append(l[1])
            _type2name[l[1]] = l[3]
    return _concept2type, _type2name


def update_type_map_w_hierarchy(args, concept2type):
    print('Reading in type hierarchy -> updating concept to type map')
    # get the type isa hierarchy
    type_isa_map = defaultdict(list)
    with open(args.type_hierarchy_file, 'r') as f:
        for line in f:
            typeA, relation, typeB, _ = line.strip().split('|')
            if relation == 'T186':    # isa relation encoding
                type_isa_map[typeA].append(typeB)

    # update the concept2type map (one -> many) with type hierarchy
    _expanded_concept2type = defaultdict(list)
    for concept, types in concept2type.items():
        _expanded_types = copy.deepcopy(types)
        for t in types:
            _expanded_types += type_isa_map[t]
        _expanded_types = list(set(_expanded_types))
        _expanded_concept2type[concept] = _expanded_types

    return _expanded_concept2type


def read_untyped_concept_map(args):
    print('Reading in untyped concepts -> full name to concept map')
    _name2concept = defaultdict(list)

    if os.path.isfile(args.name2concept_binary):
        with open(args.name2concept_binary, 'rb') as f:
            _name2concept = pickle.load(f)
    else:
        with open(args.concept_file, 'r') as f:
            reader = csv.reader(f, delimiter='|')
            for l in tqdm(reader):
                normalized_name = normalize_string(l[14])
                concept_id = l[0]
                type_ids = concept2type[concept_id] # concept can be of multiple types
                if normalized_name:
                    for type_id in type_ids:
                        if type_id in st21pv_types:
                            _name2concept[normalized_name].append(concept_id)
                            #_name2concept[l[14]].append(concept_id)

        with open(args.name2concept_binary, 'wb') as f:
            pickle.dump(_name2concept, f, pickle.HIGHEST_PROTOCOL)

    _concept2name = _read_concept2name(args)
    return _name2concept, _concept2name

def read_typed_concept_map(args, types, concept2type):
    print('Reading in typed concepts -> full name to concept map')
    _name2concept = {t : defaultdict(list) for t in types}
    if os.path.isfile(args.name2concept_binary):
        with open(args.name2concept_binary, 'rb') as f:
            _name2concept = pickle.load(f)
    else:
        with open(args.concept_file, 'r') as f:
            reader = csv.reader(f, delimiter='|')
            for l in tqdm(reader):
                normalized_name = normalize_string(l[14])
                concept_id = l[0]
                type_ids = concept2type[concept_id] # concept can be of multiple types
                if normalized_name:
                    for t in type_ids:
                        _name2concept[t][normalized_name].append(concept_id)
        with open(args.name2concept_binary, 'wb') as f:
            pickle.dump(_name2concept, f, pickle.HIGHEST_PROTOCOL)

    _concept2name = _read_concept2name(args)
    return _name2concept, _concept2name


def _read_concept2name(args):
    print('Reading in concepts -> full concept to name map')
    _concept2name = defaultdict(list)
    if os.path.isfile(args.concept2name_binary):
        with open(args.concept2name_binary, 'rb') as f:
            _concept2name = pickle.load(f)
    else:
        with open(args.concept_file, 'r') as f:
            reader = csv.reader(f, delimiter='|')
            for l in tqdm(reader):
                name = l[14].strip()
                concept_id = l[0]
                _concept2name[concept_id].append(name)
        for concept in tqdm(_concept2name.keys()):
            _concept2name[concept].sort(key=Counter(_concept2name[concept]).get, reverse=True)
            _concept2name[concept] = set(_concept2name[concept])
        with open(args.concept2name_binary, 'wb') as f:
            pickle.dump(_concept2name, f, pickle.HIGHEST_PROTOCOL)
    return _concept2name


def read_typed_mentions(mention_file, abbrev_map, name2concept, type2name):
    '''
    read in pubtator formatted data of entity mentions
    :param mention_file:
    :param abbrev_map: mappings from abbreviated doc mentions to full names
    :param name2concept: mappings from full names to entity id's
    :param type2name: mappings from type id's to full names of types
    :return: list of mentions names and corresponding entity label annotations
    '''
    type_mentions = {t: defaultdict(list) for t in type2name.keys()}

    print ('Reading mentions from %s' % mention_file)
    with (gzip.open(mention_file, 'rt') if mention_file.endswith('gz') else open(mention_file, 'r')) as in_f:
        n = 0
        for l in tqdm(in_f):
            parts = l.strip().split('\t')
            if len(parts) == 6:
                doc_id, start, end, mention, e_types, e_id = parts
                n += 1
                e_id = e_id.replace('MESH:', '').replace('OMIM:', '').replace('UMLS:', '')
                # for gene vocabulary (if multiple ';' seperated ids, take the first)
                e_id = re.sub('\(Tax:[0-9]*\)', '', e_id).split(';')[0]
                mention = '%s\t%s' % (doc_id, mention)
                if mention in abbrev_map:
                    norm_mention = normalize_string(abbrev_map[mention])
                else:
                    norm_mention = normalize_string(mention)
                for _e_type in e_types.strip().split(','):
                    type_mentions[_e_type][norm_mention].append((e_id, doc_id, mention))
    print('Read in %d mentions' % n)
    return type_mentions

def read_untyped_mentions(mention_file, abbrev_map, name2concept):
    '''
    read in pubtator formatted data of entity mentions
    :param mention_file:
    :param abbrev_map: mappings from abbreviated doc mentions to full names
    :param name2concept: mappings from full names to entity id's
    :return: list of mentions names and corresponding entity label annotations
    '''
    _mentions = defaultdict(list)

    print ('Reading mentions from %s' % mention_file)
    with (gzip.open(mention_file, 'rt') if mention_file.endswith('gz') else open(mention_file, 'r')) as in_f:
        n = 0
        for l in tqdm(in_f):
            parts = l.strip().split('\t')
            if len(parts) == 6:
                doc_id, start, end, mention, e_types, e_id = parts
                n += 1
                e_id = e_id.replace('MESH:', '').replace('OMIM:', '').replace('UMLS:', '')
                # for gene vocabulary (if multiple ';' seperated ids, take the first)
                e_id = re.sub('\(Tax:[0-9]*\)', '', e_id).split(';')[0]
                mention = '%s\t%s' % (doc_id, mention)
                if mention in abbrev_map:
                    norm_mention = normalize_string(abbrev_map[mention])
                else:
                    norm_mention = normalize_string(mention)
                _mentions[norm_mention].append((e_id, doc_id, mention))
    print('Read in %d mentions' % n)
    return _mentions


def vector_compare(mentions, kg_entities, char_vectorizer, word_vectorizer,
                   skipgram_model, out_file_prefix, num_shards, e_type, concept2name,
                   candidates_by_doc):
    '''
    convert mention strings and kg entity strings to tfidf vectors and get topk candidates for each mention
    :param word_vectorizer:
    :param char_vectorizer:
    :param mention_names: mention strings
    :param kg_names: kg entity strings
    :param mention_labels: mention entity labels
    :param kg_labels: kg entity labels
    :return:
    '''

    # must use either skipgram model or tfidf vectorizer
    assert (char_vectorizer == None and word_vectorizer == None) != (skipgram_model == None)

    tfidf_vectors = True if skipgram_model == None else False

    print('Creating empty hits and misses files')
    for topK in {1, 10, 25, 100, 250, args.num_candidates}:
        with open('.'.join([out_file_prefix, str(topK), 'hits']), 'w') as hit_file,\
             open('.'.join([out_file_prefix, str(topK), 'misses']), 'w') as miss_file:
            hit_file.write('')
            miss_file.write('')

    print('Creating header for candidates files')
    out_str = '\t'.join(['normalized_mention',
                         'mention',
                         'gold entity',
                         'candidate',
                         'candidate_score', 
                         'entities referenced by candidate',
                         'mention strings used to refer to ground truth entity'])
    with open('.'.join([out_file_prefix, 'candidates.hit.tsv']), 'w') as f:
        f.write(out_str + '\n')
    with open('.'.join([out_file_prefix, 'candidates.miss.tsv']), 'w') as f:
        f.write(out_str + '\n')

    print('Converting kg entity names to vectors')
    kg_names = list(kg_entities.keys())
    kg_labels = list(kg_entities.values())

    def get_skipgram_vectors(strings):
        _vectors = []
        for s in strings:
            _vectors.append(skipgram_model[s])
        return np.vstack(_vectors)

    if tfidf_vectors:
        kg_vectors = sparse.hstack((char_vectorizer.transform(kg_names),
                                   word_vectorizer.transform(kg_names))).tocsr()
    else:
        kg_vectors = get_skipgram_vectors(kg_names)

    mention_names = list(mentions.keys())

    # shard pairwise distance to not run out of memory
    shard_start = 0
    delta = max(1, int(len(mention_names) / num_shards))
    topk_hits = defaultdict(int)
    topk_misses = defaultdict(int)
    max_few_misses = 10
    for shard_num in range(num_shards):
        sys.stdout.write('\rshard: %d\033[K' % shard_num)
        shard_end = shard_start + delta if shard_num < (num_shards-1) else len(mention_names)
        mention_name_shard = mention_names[shard_start:shard_end]
        sys.stdout.write(' Converting text mentions to vectors... ')
        if tfidf_vectors:
            mention_vectors = sparse.hstack((char_vectorizer.transform(mention_name_shard),
                                             word_vectorizer.transform(mention_name_shard))).tocsr()
        else:
            mention_vectors = get_skipgram_vectors(mention_name_shard)
        sys.stdout.write(' Getting pairwise scores... ')
        pair_scores = pairwise_distances(mention_vectors, kg_vectors, metric='cosine', n_jobs=10)
        sys.stdout.write(' sorting pairwise scores... ')
        all_arg_mins = np.argsort(pair_scores)

        for row in range(pair_scores.shape[0]):
            sys.stdout.write('\rrow: %d' % row)
            sys.stdout.flush()
            arg_mins = all_arg_mins[row]
            # current text mention
            normalized_mention = mention_names[shard_start+row]
            # TODO, only taking first entity that has this normalized mention but might need to deal with all
            gold_e_id, doc_id, mention = mentions[normalized_mention][0]
            for topK in {1, 10, 25, 100, 250, args.num_candidates}:
                # find topk most similar kg entity names to this text mention
                i, k = 0, 0
                hit = False
                used_entities = set()
                top_few_misses = []
                top_hit = ''

                max_match_idx = arg_mins[i]
                with open('.'.join([out_file_prefix, str(topK), 'hits']), 'a') as hit_file,\
                     open('.'.join([out_file_prefix, str(topK), 'misses']), 'a') as miss_file:
                    # look at topk candidates, exit early if true entity found except when exporting
                    while k < topK and i < arg_mins.shape[0] and (topK == args.num_candidates or not hit):
                        # index of current kg entity
                        match_idx = arg_mins[i]
                        for kg_entity in kg_labels[match_idx]:
                            # the same entity can have multiple kb names that we compare against
                            if kg_entity not in used_entities:
                                k += 1
                                used_entities.add(kg_entity)
                                candidates_by_doc[doc_id].add(kg_entity)
                                # if this is the gold query entity, update hit count
                                if kg_entity == gold_e_id:
                                    hit = True
                                    max_match_idx = match_idx
                                    top_hit = '\t'.join([doc_id,
                                                         normalized_mention,
                                                         mention,
                                                         gold_e_id,
                                                         kg_names[match_idx],
                                                         str(pair_scores[row][match_idx]), 
                                                         ', '.join(kg_labels[match_idx]),
                                                         ', '.join(list(concept2name[gold_e_id]))])
                                    break
                                elif len(top_few_misses) <= max_few_misses:
                                    miss_str = '\t'.join([doc_id,
                                                          normalized_mention,
                                                          mention,
                                                          gold_e_id,
                                                          kg_names[match_idx],
                                                          str(pair_scores[row][match_idx]), 
                                                          ', '.join(kg_labels[match_idx]),
                                                          ', '.join(list(concept2name[gold_e_id]))])
                                    top_few_misses.append(miss_str)
                        i += 1
                    # was correct entity within the topk candidates?
                    if hit:
                        topk_hits[topK] += len(mentions[normalized_mention])
                        for _, _, m in mentions[normalized_mention]:
                            out_line = '%s\t%s\t%s\t%s\t%s\t%1.8f\t%s\t%s\n' \
                                       % (doc_id, normalized_mention, m, e_type,
                                          gold_e_id,
                                          (1.0 - pair_scores[row][max_match_idx]),
                                          "|".join(concept2name[gold_e_id]),
                                          "|".join(concept2name[kg_labels[max_match_idx][0]]))
                            hit_file.write(out_line)
                    else:
                        topk_misses[topK] += len(mentions[normalized_mention])
                        for _, _, m in mentions[normalized_mention]:
                            out_line = '%s\t%s\t%s\t%s\t%s\t%1.8f\t%s\t%s\n' \
                                       % (doc_id, normalized_mention, m, e_type,
                                          gold_e_id,
                                          (1.0 - pair_scores[row][max_match_idx]),
                                          "|".join(concept2name[gold_e_id]),
                                          "|".join(concept2name[kg_labels[max_match_idx][0]]))
                            miss_file.write(out_line)

                if topK == max_few_misses:
                    if not hit:
                        with open('.'.join([out_file_prefix, 'candidates.miss.tsv']), 'a') as f:
                            for miss_str in top_few_misses:
                                f.write(miss_str + '\n')
                    else:
                        with open('.'.join([out_file_prefix, 'candidates.hit.tsv']), 'a') as f:
                            f.write(top_hit + '\n')

        shard_start = shard_end

    for topK in sorted(topk_misses.keys()):
        print('\nk: %d hits: %d\t misses: %d \t hit rate %2.2f'
              % (topK, topk_hits[topK], topk_misses[topK],
                 (100 * topk_hits[topK] / (1 + topk_hits[topK] + topk_misses[topK]))))


def compute_top_k_indices(args, mention_vectors, mention_shard_num,
                          entity_names, num_entity_shards, entity_shard_num,
                          max_top_k):
    entity_shard_delta = max(1, int(len(entity_names) / num_entity_shards))
    entity_shard_start = entity_shard_num * entity_shard_delta
    entity_shard_end = entity_shard_start + entity_shard_delta \
                        if entity_shard_num < (num_entity_shards-1)\
                        else len(entity_names)
    entity_name_shard = entity_names[entity_shard_start:entity_shard_end]
    entity_vectors = sparse.hstack((char_vectorizer.transform(entity_name_shard),
                                    word_vectorizer.transform(entity_name_shard))).tocsr()

    embed()
    exit()

    cp = ci.MultiClusterIndex(entity_vectors, range(len(entity_name_shard)))
    start_time = time.time()

    top_k_indices = cp.search(mention_vectors, k=1, k_clusters=1, return_distance=False)
    top_k_indices = np.asarray(top_k_indices, dtype=np.int32)

    print('Time one shard pair - fast sparse kNN: ', time.time() - start_time)
    start_time = time.time()

    pair_scores = pairwise_distances(mention_vectors, entity_vectors, metric='cosine', n_jobs=1)
    top_k_indices = np.argpartition(pair_scores, max_top_k)[:,:max_top_k]

    print('Time one shard pair - exact pair scores: ', time.time() - start_time)
    exit()

    #pair_scores = pairwise_distances(mention_vectors, entity_vectors, metric='cosine', n_jobs=10)
    #top_k_indices = np.argpartition(pair_scores, max_top_k)[:,:max_top_k]

    #out_fname = args.out_file_prefix + 'top_k_indices_{}-{}'.format(mention_shard_num,
    #                                                           entity_shard_num)
    np.savez(out_fname, top_k_indices)


def get_candidates(args, mentions, name2concept, char_vectorizer,
                   word_vectorizer, num_mention_shards, num_entity_shards,
                   max_top_k):
    mention_names = list(mentions.keys())
    entity_names = list(name2concept.keys())

    mention_shard_start = 0
    mention_shard_delta = max(1, int(len(mention_names) / num_mention_shards))
    for mention_shard_num in trange(num_mention_shards, desc='Mention shard: ', leave=True):
        mention_shard_end = mention_shard_start + mention_shard_delta \
                            if mention_shard_num < (num_mention_shards-1)\
                            else len(mention_names)
        mention_name_shard = mention_names[mention_shard_start:mention_shard_end]
        mention_vectors = sparse.hstack((char_vectorizer.transform(mention_name_shard),
                                         word_vectorizer.transform(mention_name_shard))).tocsr()
        

        for entity_shard_num in trange(num_entity_shards, desc='Entity shard: '):
            compute_top_k_indices(args,
                                  mention_vectors,
                                  mention_shard_num,
                                  entity_names,
                                  num_entity_shards,
                                  entity_shard_num,
                                  max_top_k)




        #Parallel(n_jobs=2, max_nbytes=None)(
        #        delayed(compute_top_k_indices)(
        #                copy.deepcopy(args),
        #                copy.deepcopy(mention_vectors),
        #                copy.deepcopy(mention_shard_num),
        #                copy.deepcopy(entity_names),
        #                num_entity_shards,
        #                entity_shard_num,
        #                max_top_k
        #            ) for entity_shard_num in range(2))



        mention_shard_start = mention_shard_end



    #print('Compute mention vectors')
    #mention_vectors = sparse.hstack((char_vectorizer.transform(mention_names),
    #                                 word_vectorizer.transform(mention_names))).tocsr()
    #print('Compute kg vectors')
    #kg_vectors = sparse.hstack((char_vectorizer.transform(kg_names),
    #                            word_vectorizer.transform(kg_names))).tocsr()

    embed()
    exit()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    abbrev_map = read_abbrev_map(args)

    concept2type, type2name = read_type_map(args)
    concept2type = update_type_map_w_hierarchy(args, concept2type)
    types = list(type2name.keys())
    name2concept, concept2name = read_typed_concept_map(args, types, concept2type)
    
    #name2concept, concept2name = read_untyped_concept_map(args)

    embed()
    exit()
    candidates = {str(k) : defaultdict(set) for k in {1, 10, 25, 100, 250, args.num_candidates}}

    print('Reading in text mention data')
    train_mentions = read_untyped_mentions(args.train_file, abbrev_map, name2concept)
    dev_mentions = read_untyped_mentions(args.dev_file, abbrev_map, name2concept)
    test_mentions = read_untyped_mentions(args.test_file, abbrev_map, name2concept)

    all_names = list(train_mentions.keys()) + list(name2concept.keys())
    word_grams = [int(x) for x in args.word_grams.split(',')]
    char_grams = [int(x) for x in args.char_grams.split(',')]
    
    # Character n-gram vectorizer
    if os.path.isfile(args.char_vec_binary):
        print('Loading char vectorizer')
        with open(args.char_vec_binary, 'rb') as f:
            char_vectorizer = pickle.load(f)
    else:
        print('Learning char vectorizer')
        char_vectorizer = TfidfVectorizer(analyzer='char',
                                          ngram_range=char_grams,
                                          max_features=args.max_features,
                                          stop_words='english')
        char_vectorizer.fit(all_names)
        with open(args.char_vec_binary, 'wb') as f:
            pickle.dump(char_vectorizer, f, pickle.HIGHEST_PROTOCOL)

    # Word n-gram vectorizer
    if os.path.isfile(args.word_vec_binary):
        print('Loading word vectorizer')
        with open(args.word_vec_binary, 'rb') as f:
            word_vectorizer = pickle.load(f)
    else:
        print('Learning word vectorizer')
        word_vectorizer = TfidfVectorizer(analyzer='word',
                                          ngram_range=word_grams,
                                          max_features=args.max_features,
                                          tokenizer=genia_tokenizer.tokenize)
        word_vectorizer.fit(all_names)
        with open(args.word_vec_binary, 'wb') as f:
            pickle.dump(word_vectorizer, f, pickle.HIGHEST_PROTOCOL)

    print('Generating candidates')
    get_candidates(args, train_mentions, name2concept, char_vectorizer,
                   word_vectorizer, 100, 1000, 250)

    exit()

    ##########################################################################
    tfidf_vectors = True
    if not tfidf_vectors:
        print('Loading skipgram model')
        skipgram_model = fasttext.load_model(args.skipgram_lm_file)

    def link_mentions(e_type):
        train_mentions = train_type_mentions[e_type]
        dev_mentions = dev_type_mentions[e_type]
        test_mentions = test_type_mentions[e_type]

        if not train_mentions and not dev_mentions and not test_mentions:
            print('No mentions of type: %s' % type2name[e_type])
            return

        if tfidf_vectors:
            print('Vectoring %s' % type2name[e_type])
            all_names = list(train_mentions.keys()) + list(name2concept[e_type].keys())
            word_grams = [int(x) for x in args.word_grams.split(',')]
            char_grams = [int(x) for x in args.char_grams.split(',')]
            
            print('[%s] fitting tfidf vectorizers' % (type2name[e_type]))
            word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=word_grams,
                                              max_features=args.max_features, tokenizer=genia_tokenizer.tokenize)
            char_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=char_grams,
                                              max_features=args.max_features, stop_words='english')
            word_vectorizer.fit(all_names)
            char_vectorizer.fit(all_names)

        splits = ['train', 'dev', 'test']
        mentions_list = [train_mentions, dev_mentions, test_mentions]
        for split, mentions in zip(splits, mentions_list):
            if mentions:
                print('[%s] Using %s mentions' % (type2name[e_type], split))
                if tfidf_vectors: 
                    vector_compare(mentions, name2concept[e_type],
                                   char_vectorizer, word_vectorizer, None,
                                   args.out_file_prefix + type2name[e_type].replace(' ', '_') + "." + split,
                                   args.num_shards, e_type, concept2name,
                                   candidates[e_type])
                else:
                    assert False
                    vector_compare(mentions, name2concept[e_type],
                                   None, None, skipgram_model,
                                   args.out_file_prefix + type2name[e_type].replace(' ', '_') + "." + split,
                                   args.num_shards, e_type, concept2name,
                                   candidates[e_type])
            else:
                print('[%s] No %s mentions' % (type2name[e_type], split))

    for e_type in type2name.keys():
        link_mentions(e_type)

    with open(args.out_file_prefix + 'candidate_entities.pkl', 'wb') as f:
        pickle.dump(candidates, f, pickle.HIGHEST_PROTOCOL)
