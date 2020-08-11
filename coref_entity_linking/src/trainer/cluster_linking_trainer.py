from collections import defaultdict
import logging
from functools import reduce
import numpy as np
import os
import pickle
from scipy.sparse import coo_matrix
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm, trange
import random

from clustering import (TripletDatasetBuilder,
                        SoftmaxDatasetBuilder,
                        ThresholdDatasetBuilder)
from data.datasets import MetaClusterDataset, InferenceEmbeddingDataset
from data.dataloaders import (MetaClusterDataLoader,
                              InferenceEmbeddingDataLoader)
from evaluation.evaluation import eval_wdoc, eval_xdoc
from model import MirrorEmbeddingModel, VersatileModel
from trainer.trainer import Trainer
from trainer.emb_sub_trainer import EmbeddingSubTrainer
from trainer.concat_sub_trainer import ConcatenationSubTrainer
from utils.comm import (all_gather,
                        broadcast,
                        get_world_size,
                        get_rank,
                        synchronize)
from utils.knn_index import WithinDocNNIndex, CrossDocNNIndex
from utils.misc import flatten, unique, dict_merge_with

if get_rank() == 0:
    import wandb

from IPython import embed


logger = logging.getLogger(__name__)


class ClusterLinkingTrainer(Trainer):

    def __init__(self, args):
        super(ClusterLinkingTrainer, self).__init__(args)

        args.num_entities = self.train_metadata.num_entities

        # create sub-trainers for models
        self.create_sub_trainers()

        # create knn_index and supervised clustering dataset builder
        if args.do_train or args.do_train_eval:
            #self.create_knn_index('train')
            if args.do_train:
                if 'triplet' in args.training_method:
                    self.dataset_builder = TripletDatasetBuilder(args)
                elif args.training_method == 'softmax':
                    self.dataset_builder = SoftmaxDatasetBuilder(args)
                elif args.training_method == 'threshold':
                    self.dataset_builder = ThresholdDatasetBuilder(args)
                else:
                    raise ValueError('unsupported training_method')
        if args.do_val: # or args.evaluate_during_training:
            pass
            #self.create_knn_index('val')
        if args.do_test:
            pass
            #self.create_knn_index('test')
    
    def create_models(self):
        logger.info('Creating models.')
        args = self.args
        self.models = {
                'affinity_model': VersatileModel(
                    args,
                    name='affinity_model'
                )
        }
        args.tokenizer = self.models['affinity_model'].tokenizer

    def create_sub_trainers(self):
        logger.info('Creating sub-trainers.')
        args = self.args
        for name, model in self.models.items():
            optimizer = self.optimizers[name] if args.do_train else None
            scheduler = self.schedulers[name] if args.do_train else None
            if isinstance(model.module, VersatileModel):
                self.embed_sub_trainer = EmbeddingSubTrainer(
                        args,
                        model,
                        optimizer,
                        scheduler
                )
                self.concat_sub_trainer = ConcatenationSubTrainer(
                        args,
                        model,
                        optimizer,
                        scheduler
                )
            else:
                raise ValueError('module not supported by a sub trainer')
    
    def create_train_dataloader(self):
        args = self.args

        # load and cache examples and get the metadata for the dataset
        self.load_and_cache_examples(split='train', evaluate=False)

        # determine the set of gold clusters depending on the setting
        if args.clustering_domain == 'within_doc':
            clusters = flatten([list(doc.values()) 
                    for doc in self.train_metadata.wdoc_clusters.values()])
        elif args.clustering_domain == 'cross_doc':
            clusters = list(self.train_metadata.xdoc_clusters.values())
        else:
            raise ValueError('Invalid clustering_domain')

        logger.warning('!!!! Using reduced dataset intended for tiny experiment !!!!')
        args._tiny_exp_clusters = clusters[:300]
        args._tiny_exp_examples = flatten(args._tiny_exp_clusters)

        #self.train_dataset = MetaClusterDataset(clusters)
        self.train_dataset = MetaClusterDataset(args._tiny_exp_clusters)
        self.train_dataloader = MetaClusterDataLoader(
                args, self.train_dataset)

    def create_train_eval_dataloader(self):
        args = self.args

        # load and cache examples and get the metadata for the dataset
        self.load_and_cache_examples(split='train', evaluate=True)

        if args.available_entities in ['candidates_only', 'knn_candidates']:
            examples = flatten([[k] + v
                    for k, v in self.train_metadata.midx2cand.items()])
            examples.extend(self.train_metadata.midx2eidx.values())
        elif args.available_entities == 'open_domain':
            examples = list(self.train_metadata.idx2uid.keys())
        else:
            raise ValueError('Invalid available_entities')
        examples = unique(examples)

        logger.warning('!!!! Using reduced dataset intended for tiny experiment !!!!')
        args._tiny_exp_clusters=[[2327296, 1501090, 2327266, 2327239, 2327274, 2327281, 2327250, 2327255, 2327288], [809089, 2327300, 2327240, 2327273, 2327242, 2327245, 2327277, 2327279, 2327280, 2327251, 2327283, 2327286, 2327287, 2327257, 2327258, 2327259, 2327294, 2327291], [2327268, 2327301, 2327241, 2327293, 2327243, 2327276, 2327278, 2327252, 2327285, 762135, 2327290, 2327261], [2327244, 763614], [637320, 2327246], [662464, 2327247], [2327248, 1053329], [2327297, 2327298, 2327267, 2327275, 2327249, 2327282, 2327284, 2327256, 2327289, 1215420], [764923, 2327253], [1053165, 2327254], [2327260, 1119724], [1112971, 2327262], [2327263, 1052319], [2327264, 1053391], [2327265, 382702], [2327269, 1112573], [440169, 2327270], [813413, 2327271], [17912, 2327272], [763889, 2327292, 2327295], [805441, 2327299], [2327309, 1127565, 2327302], [2327303, 2327317, 631375], [2327304, 1053678], [1074984, 2327305], [2327306, 943839], [2327307, 604147], [1072316, 2327308], [428810, 2327310], [2327311, 367847], [2327312, 1073839], [2327313, 610542], [2327314, 1074007], [2327315, 1074422], [2327316, 1053165], [2327360, 2327330, 556644, 2327336, 2327342, 2327318, 2327354, 2327359], [2327355, 2327361, 2327331, 771460, 2327333, 2327338, 2327345, 2327319, 2327321, 2327322, 2327323, 2327327], [2327332, 2327337, 2327343, 1073842, 2327320, 2327353, 2327324, 2327358], [975554, 2327325], [2327362, 2327365, 552445, 2327326], [2327328, 1119799], [2327329, 368613, 2327352, 2327356, 2327357], [368962, 2327334], [2327351, 2327335, 2327350, 1072335], [2327349, 2327339, 372237], [2327340, 1119725], [552219, 2327341], [2327344, 1077386], [371616, 2327346], [385272, 2327347], [2327348, 423742], [552288, 2327363], [1119258, 2327364], [555601, 2327366, 2327375], [2327367, 938985, 2327371, 2327373, 2327380], [1092296, 2327368, 2327382], [2327369, 1071921, 2327383], [1052329, 2327370], [1071763, 2327372], [2327392, 2327397, 2327374, 1112558, 2327381, 2327388, 2327390, 2327391], [2327376, 987046], [1052552, 2327377], [2327378, 1511605], [1119248, 2327379], [2327384, 2327389, 1119725], [2327393, 2327395, 2327396, 2327385, 1003388], [2327386, 1099399], [2327394, 2327387, 1073854], [560875, 2327413, 2327398], [1052318, 2327399], [2327425, 2327400, 2327402, 2327403, 2327404, 2327407, 943514], [2327401, 1119799], [2327405, 1507806], [1118729, 2327406], [983128, 2327408], [2327409, 988234], [2327410, 987301], [2327419, 2327411, 1012828], [2327412, 2327420, 1112524], [1003388, 2327414], [803548, 2327415], [2327416, 553185], [2327417, 1507660], [589352, 2327418], [939416, 2327424, 2327421], [2327422, 1118311], [1524766, 2327423], [555721, 2327426], [2327427, 2327432, 2327434, 1124942, 2327446, 2327454], [2327428, 1536988], [1505314, 2327459, 2327429], [2327430, 2327463, 1150343, 2327433, 2327444], [2327431, 2327464, 2327465, 1503706, 2327452, 2327450], [1552761, 2327435], [2327447, 2327436, 1540943], [2327449, 1585323, 2327437], [2327448, 1549957, 2327438], [2327439, 2327458, 1503759], [2327440, 2327457, 1505853], [2327456, 2327441, 1503771], [2327442, 1503700, 2327461], [1504194, 2327443, 2327462], [1549853, 2327445], [960984, 2327451, 2327453], [2327455, 1503727], [2327460, 1503726], [2327466, 989514], [2327489, 2327467, 2327469, 2327474, 2327485, 382975], [2327488, 2327468, 2327470, 2327476, 511094], [1018622, 2327471], [2327472, 1052333], [2327490, 982161, 2327473, 2327477, 2327479], [1072297, 2327475], [761504, 2327491, 2327478], [2327480, 553338], [2327481, 2327487, 617703], [2327482, 1004806], [555721, 2327483], [2327484, 762309], [1072337, 2327486], [2327492, 566711], [367876, 2327493], [2327520, 2327524, 1342821, 2327494, 2327528, 2327535, 2327510], [803491, 2327495, 2327499, 2327536, 2327514], [2327525, 2327496, 2327529, 2327498, 624202, 2327503, 2327504, 2327538, 2327518], [2327497, 1053165], [1119258, 2327500], [1395457, 2327501], [1052245, 2327502], [2327505, 1527717], [2327506, 761644], [550554, 2327507], [2327508, 985431], [975433, 2327509], [956230, 2327511], [2327512, 1013833, 2327521], [1148369, 2327513], [940410, 2327515], [2327516, 2327532, 597031], [2327526, 2327530, 2327533, 1119799, 2327517, 2327519], [2327522, 2327531, 1529405, 2327527], [1527674, 2327523], [555721, 2327534], [2327537, 1119308], [940423, 2327562, 2327594, 2327566, 2327539, 2327576], [2327553, 2327559, 2327596, 2327571, 2327540, 2327545, 1088412], [2327554, 1079172, 2327560, 2327597, 2327601, 2327541], [2327555, 2327561, 2327598, 376529, 2327542, 2327546], [552445, 2327586, 2327557, 2327564, 2327600, 2327603, 2327543, 2327579, 2327549, 2327583], [2327556, 764167, 2327563, 2327595, 2327567, 2327544], [771035, 2327547], [2327570, 765236, 2327548, 2327551], [2327550, 941119], [2327552, 553339], [1052323, 2327558], [2327588, 2327565, 2327604, 574799], [2327568, 940023], [2327569, 1053663], [2327572, 368758], [2327592, 368369, 2327573], [368962, 2327574], [2327585, 2327587, 2327575, 1003388, 2327582], [2327577, 1055420], [2327578, 1032523], [1518042, 2327580, 2327590], [2327593, 633764, 2327581], [2327584, 555721], [1534273, 2327589], [1105081, 2327591], [2327599, 589775], [2327602, 762532], [2327616, 2327633, 2327605, 2327613, 939998, 2327615], [2327606, 2327611, 845998], [2327617, 797155, 2327607, 2327612, 2327614], [2327608, 960754], [2327609, 1013490], [2327610, 949915], [2327618, 1112573], [2327619, 1052303], [1050921, 2327620], [2327621, 2327622, 2327624, 2327625, 2327630, 2327631, 1119799], [995832, 2327623], [2327626, 765806], [2327627, 1510437], [989704, 2327628], [2327629, 762590], [2327632, 633738], [2327634, 582115], [958920, 2327635, 2327661], [2327660, 1507730, 2327636], [2327637, 552559], [2327642, 1079332, 2327638, 2327662], [1508060, 2327639], [1507744, 2327643, 2327640, 2327653], [2327641, 1053165, 2327663], [2327644, 1071757], [2327645, 1072191], [1071713, 2327646], [997194, 2327647], [1079664, 2327648], [2327649, 762158], [618410, 2327650], [2327651, 1074022], [553185, 2327652], [2327657, 1198747, 2327654], [1074266, 2327658, 2327655], [1071856, 2327656], [2327659, 1074820], [438728, 2327722, 2327664, 2327711], [2327665, 439682, 2327715, 2327727], [2327714, 2327719, 2327691, 2327723, 2327728, 2327666, 762774, 2327710], [2327713, 2327688, 2327693, 2327725, 762703, 2327667, 2327702], [2327716, 2327695, 2327730, 2327668, 763159, 2327705], [2327712, 763717, 2327692, 2327724, 2327697, 2327729, 2327669, 2327670, 2327676], [762285, 2327671], [2327672, 763910], [2327673, 2327677, 596837, 2327701], [2327674, 1111716], [2327675, 1111750], [368538, 2327678], [552562, 2327679], [765600, 2327680], [552369, 2327681], [552178, 2327682], [552289, 2327683], [2327684, 552916], [761461, 2327685], [552411, 2327686], [765129, 2327687], [2327689, 1527131], [2327690, 1053165], [762329, 2327721, 2327708, 2327694], [2327696, 1119297], [2327698, 1053292], [2327720, 2327699, 368246], [2327700, 1127356], [2327706, 815428, 2327703], [779712, 2327707, 2327704], [552288, 2327709], [673656, 2327717], [761344, 2327718], [556261, 2327726], [648169, 2327731, 2327734, 2327739, 2327743], [2327732, 652205, 2327742, 2327735], [1507897, 2327733], [2327736, 1112869], [552288, 2327737, 2327744], [1003865, 2327738], [2327740, 1119799], [1114417, 2327741], [2327745, 2327755, 761724, 2327751], [2327746, 988390], [939218, 2327747], [2327748, 367847], [2327749, 941062], [941476, 2327750], [2327752, 765171], [2327753, 773918], [2327754, 1114549], [556889, 2327756], [773833, 2327757], [555693, 2327758], [553441, 2327759], [2327760, 2327800, 1075908], [2327761, 1079438, 2327767], [2327762, 1072363, 2327799], [2327763, 1536821], [1085458, 2327764], [1512804, 2327782, 2327787, 2327788, 2327791, 2327765, 2327798, 2327775], [2327781, 1071856, 2327766, 2327769, 2327771, 2327773, 2327774], [2327768, 2327785, 1537428, 2327796], [2327770, 1072314], [2327783, 2327789, 1072337, 2327794, 2327772], [2327776, 1536890], [2327777, 1536838], [2327778, 1537139], [1550099, 2327779], [2327793, 1074818, 2327795, 2327780], [2327784, 1072338], [1537136, 2327786], [1549840, 2327790], [2327792, 1085459], [2327797, 1075767], [2327801, 1260059], [2327802, 850132], [2327810, 2327813, 2327846, 2327820, 438872, 2327803], [2327844, 2327814, 2327823, 1124975, 2327826, 2327829, 2327804], [2327840, 2327845, 2327815, 2327824, 2327827, 2327837, 1158493, 2327805], [2327808, 2327811, 2327847, 2327816, 2327821, 763128, 2327806], [2327832, 555721, 2327807], [2327809, 1156381], [2327812, 1053165], [2327817, 955619], [2327818, 991143], [2327819, 572014], [939264, 2327822], [2327825, 1510646], [1052256, 2327828], [2327841, 2327835, 1214445, 2327830], [1125312, 2327836, 2327831]]
        args._tiny_exp_examples=[2327296, 1501090, 2327266, 2327239, 2327274, 2327281, 2327250, 2327255, 2327288, 809089, 2327300, 2327240, 2327273, 2327242, 2327245, 2327277, 2327279, 2327280, 2327251, 2327283, 2327286, 2327287, 2327257, 2327258, 2327259, 2327294, 2327291, 2327268, 2327301, 2327241, 2327293, 2327243, 2327276, 2327278, 2327252, 2327285, 762135, 2327290, 2327261, 2327244, 763614, 637320, 2327246, 662464, 2327247, 2327248, 1053329, 2327297, 2327298, 2327267, 2327275, 2327249, 2327282, 2327284, 2327256, 2327289, 1215420, 764923, 2327253, 1053165, 2327254, 2327260, 1119724, 1112971, 2327262, 2327263, 1052319, 2327264, 1053391, 2327265, 382702, 2327269, 1112573, 440169, 2327270, 813413, 2327271, 17912, 2327272, 763889, 2327292, 2327295, 805441, 2327299, 2327309, 1127565, 2327302, 2327303, 2327317, 631375, 2327304, 1053678, 1074984, 2327305, 2327306, 943839, 2327307, 604147, 1072316, 2327308, 428810, 2327310, 2327311, 367847, 2327312, 1073839, 2327313, 610542, 2327314, 1074007, 2327315, 1074422, 2327316, 1053165, 2327360, 2327330, 556644, 2327336, 2327342, 2327318, 2327354, 2327359, 2327355, 2327361, 2327331, 771460, 2327333, 2327338, 2327345, 2327319, 2327321, 2327322, 2327323, 2327327, 2327332, 2327337, 2327343, 1073842, 2327320, 2327353, 2327324, 2327358, 975554, 2327325, 2327362, 2327365, 552445, 2327326, 2327328, 1119799, 2327329, 368613, 2327352, 2327356, 2327357, 368962, 2327334, 2327351, 2327335, 2327350, 1072335, 2327349, 2327339, 372237, 2327340, 1119725, 552219, 2327341, 2327344, 1077386, 371616, 2327346, 385272, 2327347, 2327348, 423742, 552288, 2327363, 1119258, 2327364, 555601, 2327366, 2327375, 2327367, 938985, 2327371, 2327373, 2327380, 1092296, 2327368, 2327382, 2327369, 1071921, 2327383, 1052329, 2327370, 1071763, 2327372, 2327392, 2327397, 2327374, 1112558, 2327381, 2327388, 2327390, 2327391, 2327376, 987046, 1052552, 2327377, 2327378, 1511605, 1119248, 2327379, 2327384, 2327389, 1119725, 2327393, 2327395, 2327396, 2327385, 1003388, 2327386, 1099399, 2327394, 2327387, 1073854, 560875, 2327413, 2327398, 1052318, 2327399, 2327425, 2327400, 2327402, 2327403, 2327404, 2327407, 943514, 2327401, 1119799, 2327405, 1507806, 1118729, 2327406, 983128, 2327408, 2327409, 988234, 2327410, 987301, 2327419, 2327411, 1012828, 2327412, 2327420, 1112524, 1003388, 2327414, 803548, 2327415, 2327416, 553185, 2327417, 1507660, 589352, 2327418, 939416, 2327424, 2327421, 2327422, 1118311, 1524766, 2327423, 555721, 2327426, 2327427, 2327432, 2327434, 1124942, 2327446, 2327454, 2327428, 1536988, 1505314, 2327459, 2327429, 2327430, 2327463, 1150343, 2327433, 2327444, 2327431, 2327464, 2327465, 1503706, 2327452, 2327450, 1552761, 2327435, 2327447, 2327436, 1540943, 2327449, 1585323, 2327437, 2327448, 1549957, 2327438, 2327439, 2327458, 1503759, 2327440, 2327457, 1505853, 2327456, 2327441, 1503771, 2327442, 1503700, 2327461, 1504194, 2327443, 2327462, 1549853, 2327445, 960984, 2327451, 2327453, 2327455, 1503727, 2327460, 1503726, 2327466, 989514, 2327489, 2327467, 2327469, 2327474, 2327485, 382975, 2327488, 2327468, 2327470, 2327476, 511094, 1018622, 2327471, 2327472, 1052333, 2327490, 982161, 2327473, 2327477, 2327479, 1072297, 2327475, 761504, 2327491, 2327478, 2327480, 553338, 2327481, 2327487, 617703, 2327482, 1004806, 555721, 2327483, 2327484, 762309, 1072337, 2327486, 2327492, 566711, 367876, 2327493, 2327520, 2327524, 1342821, 2327494, 2327528, 2327535, 2327510, 803491, 2327495, 2327499, 2327536, 2327514, 2327525, 2327496, 2327529, 2327498, 624202, 2327503, 2327504, 2327538, 2327518, 2327497, 1053165, 1119258, 2327500, 1395457, 2327501, 1052245, 2327502, 2327505, 1527717, 2327506, 761644, 550554, 2327507, 2327508, 985431, 975433, 2327509, 956230, 2327511, 2327512, 1013833, 2327521, 1148369, 2327513, 940410, 2327515, 2327516, 2327532, 597031, 2327526, 2327530, 2327533, 1119799, 2327517, 2327519, 2327522, 2327531, 1529405, 2327527, 1527674, 2327523, 555721, 2327534, 2327537, 1119308, 940423, 2327562, 2327594, 2327566, 2327539, 2327576, 2327553, 2327559, 2327596, 2327571, 2327540, 2327545, 1088412, 2327554, 1079172, 2327560, 2327597, 2327601, 2327541, 2327555, 2327561, 2327598, 376529, 2327542, 2327546, 552445, 2327586, 2327557, 2327564, 2327600, 2327603, 2327543, 2327579, 2327549, 2327583, 2327556, 764167, 2327563, 2327595, 2327567, 2327544, 771035, 2327547, 2327570, 765236, 2327548, 2327551, 2327550, 941119, 2327552, 553339, 1052323, 2327558, 2327588, 2327565, 2327604, 574799, 2327568, 940023, 2327569, 1053663, 2327572, 368758, 2327592, 368369, 2327573, 368962, 2327574, 2327585, 2327587, 2327575, 1003388, 2327582, 2327577, 1055420, 2327578, 1032523, 1518042, 2327580, 2327590, 2327593, 633764, 2327581, 2327584, 555721, 1534273, 2327589, 1105081, 2327591, 2327599, 589775, 2327602, 762532, 2327616, 2327633, 2327605, 2327613, 939998, 2327615, 2327606, 2327611, 845998, 2327617, 797155, 2327607, 2327612, 2327614, 2327608, 960754, 2327609, 1013490, 2327610, 949915, 2327618, 1112573, 2327619, 1052303, 1050921, 2327620, 2327621, 2327622, 2327624, 2327625, 2327630, 2327631, 1119799, 995832, 2327623, 2327626, 765806, 2327627, 1510437, 989704, 2327628, 2327629, 762590, 2327632, 633738, 2327634, 582115, 958920, 2327635, 2327661, 2327660, 1507730, 2327636, 2327637, 552559, 2327642, 1079332, 2327638, 2327662, 1508060, 2327639, 1507744, 2327643, 2327640, 2327653, 2327641, 1053165, 2327663, 2327644, 1071757, 2327645, 1072191, 1071713, 2327646, 997194, 2327647, 1079664, 2327648, 2327649, 762158, 618410, 2327650, 2327651, 1074022, 553185, 2327652, 2327657, 1198747, 2327654, 1074266, 2327658, 2327655, 1071856, 2327656, 2327659, 1074820, 438728, 2327722, 2327664, 2327711, 2327665, 439682, 2327715, 2327727, 2327714, 2327719, 2327691, 2327723, 2327728, 2327666, 762774, 2327710, 2327713, 2327688, 2327693, 2327725, 762703, 2327667, 2327702, 2327716, 2327695, 2327730, 2327668, 763159, 2327705, 2327712, 763717, 2327692, 2327724, 2327697, 2327729, 2327669, 2327670, 2327676, 762285, 2327671, 2327672, 763910, 2327673, 2327677, 596837, 2327701, 2327674, 1111716, 2327675, 1111750, 368538, 2327678, 552562, 2327679, 765600, 2327680, 552369, 2327681, 552178, 2327682, 552289, 2327683, 2327684, 552916, 761461, 2327685, 552411, 2327686, 765129, 2327687, 2327689, 1527131, 2327690, 1053165, 762329, 2327721, 2327708, 2327694, 2327696, 1119297, 2327698, 1053292, 2327720, 2327699, 368246, 2327700, 1127356, 2327706, 815428, 2327703, 779712, 2327707, 2327704, 552288, 2327709, 673656, 2327717, 761344, 2327718, 556261, 2327726, 648169, 2327731, 2327734, 2327739, 2327743, 2327732, 652205, 2327742, 2327735, 1507897, 2327733, 2327736, 1112869, 552288, 2327737, 2327744, 1003865, 2327738, 2327740, 1119799, 1114417, 2327741, 2327745, 2327755, 761724, 2327751, 2327746, 988390, 939218, 2327747, 2327748, 367847, 2327749, 941062, 941476, 2327750, 2327752, 765171, 2327753, 773918, 2327754, 1114549, 556889, 2327756, 773833, 2327757, 555693, 2327758, 553441, 2327759, 2327760, 2327800, 1075908, 2327761, 1079438, 2327767, 2327762, 1072363, 2327799, 2327763, 1536821, 1085458, 2327764, 1512804, 2327782, 2327787, 2327788, 2327791, 2327765, 2327798, 2327775, 2327781, 1071856, 2327766, 2327769, 2327771, 2327773, 2327774, 2327768, 2327785, 1537428, 2327796, 2327770, 1072314, 2327783, 2327789, 1072337, 2327794, 2327772, 2327776, 1536890, 2327777, 1536838, 2327778, 1537139, 1550099, 2327779, 2327793, 1074818, 2327795, 2327780, 2327784, 1072338, 1537136, 2327786, 1549840, 2327790, 2327792, 1085459, 2327797, 1075767, 2327801, 1260059, 2327802, 850132, 2327810, 2327813, 2327846, 2327820, 438872, 2327803, 2327844, 2327814, 2327823, 1124975, 2327826, 2327829, 2327804, 2327840, 2327845, 2327815, 2327824, 2327827, 2327837, 1158493, 2327805, 2327808, 2327811, 2327847, 2327816, 2327821, 763128, 2327806, 2327832, 555721, 2327807, 2327809, 1156381, 2327812, 1053165, 2327817, 955619, 2327818, 991143, 2327819, 572014, 939264, 2327822, 2327825, 1510646, 1052256, 2327828, 2327841, 2327835, 1214445, 2327830, 1125312, 2327836, 2327831]
        examples = args._tiny_exp_examples + list(filter(lambda x : x < self.train_metadata.num_entities, examples))
        self.train_eval_dataset = InferenceEmbeddingDataset(
                args, examples, args.train_cache_dir)
        self.train_eval_dataloader = InferenceEmbeddingDataLoader(
                args, self.train_eval_dataset)

    def create_val_dataloader(self):
        args = self.args

        # load and cache examples and get the metadata for the dataset
        self.load_and_cache_examples(split='val', evaluate=True)

        if args.available_entities in ['candidates_only', 'knn_candidates']:
            examples = flatten([[k] + v
                    for k, v in self.val_metadata.midx2cand.items()])
        elif args.available_entities == 'open_domain':
            examples = list(self.val_metadata.idx2uid.keys())
        else:
            raise ValueError('Invalid available_entities')
        examples = unique(examples)
        self.val_dataset = InferenceEmbeddingDataset(
                args, examples, args.val_cache_dir)
        self.val_dataloader = InferenceEmbeddingDataLoader(
                args, self.val_dataset)

    def create_test_dataloader(self):
        pass

    def create_knn_index(self, split=None):
        assert split == 'train' or split == 'val' or split == 'test'
        args = self.args

        NN_Index = (WithinDocNNIndex if args.clustering_domain == 'within_doc'
                        else CrossDocNNIndex)
        if split == 'train':
            self.train_knn_index = NN_Index(
                    args,
                    self.embed_sub_trainer,
                    self.train_eval_dataloader,
                    name='train'
            )
        elif split == 'val':
            self.val_knn_index = NN_Index(
                    args,
                    self.embed_sub_trainer,
                    self.val_dataloader,
                    name='val'
            )
        else:
            self.test_knn_index = NN_Index(
                    args,
                    self.embed_sub_trainer,
                    self.test_dataloader,
                    name='test'
            )

    def _build_temp_sparse_graph(self, clusters_mx, per_example_negs):
        args = self.args

        # get all of the unique examples 
        examples = clusters_mx.data.tolist()
        examples.extend(flatten(per_example_negs.tolist()))
        examples = unique(examples)
        examples = list(filter(lambda x : x >= 0, examples))

        sparse_graph = None
        if get_rank() == 0:
            ## make the list of pairs of dot products we need
            _row = clusters_mx.row
            # positives:
            local_pos_a, local_pos_b = np.where(
                    np.triu(_row[np.newaxis, :] == _row[:, np.newaxis], k=1)
            )
            pos_a = clusters_mx.data[local_pos_a]
            pos_b = clusters_mx.data[local_pos_b]
            # negatives:
            local_neg_a = np.tile(
                np.arange(per_example_negs.shape[0])[:, np.newaxis],
                (1, per_example_negs.shape[1])
            ).flatten()
            neg_a = clusters_mx.data[local_neg_a]
            neg_b = per_example_negs.flatten()

            neg_mask = (neg_b != -1)
            neg_a = neg_a[neg_mask]
            neg_b = neg_b[neg_mask]

            # create subset of the sparse graph we care about
            a = np.concatenate((pos_a, neg_a), axis=0)
            b = np.concatenate((pos_b, neg_b), axis=0)
            edges = list(zip(a, b))
            affinities = [0.0 for i, j in edges]

            # convert to coo_matrix
            edges = np.asarray(edges).T
            affinities = np.asarray(affinities)
            _sparse_num = np.max(edges) + 1
            sparse_graph = coo_matrix((affinities, edges),
                                      shape=(_sparse_num, _sparse_num))

        synchronize()
        return sparse_graph

    def train_step(self, batch):
        args = self.args

        # get the batch of clusters and approx negs for each individual example
        clusters_mx, per_example_negs = batch

        # compute scores using up-to-date model
        #sparse_graph = self.embed_sub_trainer.compute_scores_for_inference(
        #        clusters_mx, per_example_negs)
        sparse_graph = self._build_temp_sparse_graph(
                clusters_mx, per_example_negs)

        # create custom datasets for training
        embed_dataset_list = None
        concat_dataset_list = None
        dataset_metrics = None
        if get_rank() == 0:
            dataset_lists, dataset_metrics = self.dataset_builder(
                    clusters_mx, sparse_graph, self.train_metadata
            )
            embed_dataset_list, concat_dataset_list = dataset_lists
        dataset_metrics = broadcast(dataset_metrics, src=0)
        embed_dataset_list = broadcast(embed_dataset_list, src=0)
        concat_dataset_list = broadcast(concat_dataset_list, src=0)

        ## train on datasets
        #embed_return_dict = self.embed_sub_trainer.train_on_subset(
        #        embed_dataset_list, self.train_metadata
        #)

        concat_return_dict = self.concat_sub_trainer.train_on_subset(
                concat_dataset_list, self.train_metadata
        )

        #embed_return_dict = broadcast(embed_return_dict, src=0)
        concat_return_dict = broadcast(concat_return_dict, src=0)

        return_dict = {}
        return_dict.update(dataset_metrics)
        #return_dict.update(embed_return_dict)
        return_dict.update(concat_return_dict)

        #if get_rank() == 0:
        #    embed()
        #synchronize()
        #exit()

        return return_dict

    def _neg_choosing_prep(self):
        args = self.args
        metadata = self.train_metadata

        # be able to map from midx to doc and back
        self.doc2midxs = defaultdict(list)
        self.midx2doc = {}
        for doc_id, wdoc_clusters in metadata.wdoc_clusters.items():
            mentions = flatten(wdoc_clusters.values())
            mentions = [x for x in mentions if x >= metadata.num_entities]
            self.doc2midxs[doc_id] = mentions
            for midx in mentions:
                self.midx2doc[midx] = doc_id

        # need to know the available entities in this case as well
        if args.available_entities not in ['candidates_only', 'knn_candidates']:
            self.avail_entity_idxs = list(range(num_entities))

    def _choose_negs(self, batch):
        args = self.args
        negatives_list = []
        clusters = [sorted(batch.getrow(i).data.tolist())
                        for i in range(batch.shape[0])]
        for c_idxs in clusters:
            if args.clustering_domain == 'within_doc':
                # get mention idxs within document
                doc_midxs = self.doc2midxs[self.midx2doc[c_idxs[1]]]

                # produce available negative mention idxs
                neg_midxs = [m for m in doc_midxs if m not in c_idxs]

                # determine the number of mention negatives
                num_m_negs = min(args.k // 2, len(neg_midxs))

                # initialize the negs tensors
                negs = np.tile(-1, (len(c_idxs), args.k))

                if args.mention_negatives == 'context_overlap':
                    # use overlapping context negative mentions
                    neg_midxs_objects = [
                            (midx, self.train_metadata.mentions[self.train_metadata.idx2uid[midx]])
                                for midx in neg_midxs
                    ]
                    for i, idx in enumerate(c_idxs):
                        if idx >= self.train_metadata.num_entities:
                            idx_object = self.train_metadata.mentions[self.train_metadata.idx2uid[idx]]
                            neg_context_dists = [(x[0], abs(idx_object['start_index'] - x[1]['start_index']))
                                                    for x in neg_midxs_objects]
                            neg_context_dists.sort(key=lambda x : x[1])
                            local_neg_midxs, _ = zip(*neg_context_dists)
                            negs[i,:num_m_negs] = np.asarray(local_neg_midxs[:num_m_negs])
                elif args.mention_negatives == 'random':
                    for i, idx in enumerate(c_idxs):
                        if idx >= self.train_metadata.num_entities:
                            negs[i,:num_m_negs] = random.sample(neg_midxs, num_m_negs)
                else:
                    # sample mention negatives according to embedding model
                    negs[:,:num_m_negs] = self.train_knn_index.get_knn_limited_index(
                        c_idxs,
                        include_index_idxs=neg_midxs,
                        k=num_m_negs
                    )

                # produce available negative entity idxs
                # NOTE: this doesn't allow negative e-e edges (there are never any positive ones)
                num_e_negs = args.k - num_m_negs
                if args.available_entities == 'candidates_only':
                    neg_eidxs = [
                        list(filter(
                            lambda x : x != c_idxs[0],
                            self.train_metadata.midx2cand.get(i, [])
                        ))[:num_e_negs]
                            for i in c_idxs
                                if i >= self.train_metadata.num_entities
                    ]
                    neg_eidxs = [
                        l + [-1] * (num_e_negs - len(l))
                            for l in neg_eidxs
                    ]
                    negs[1:, -num_e_negs:] = np.asarray(neg_eidxs)
                else:
                    if (args.clustering_domain == 'within_doc'
                            and args.available_entities == 'knn_candidates'):
                        # custom w/in doc negative available entities
                        self.avail_entity_idxs = flatten([
                            list(filter(
                                lambda x : x != c_idxs[0],
                                self.train_metadata.midx2cand.get(i, [])
                            ))
                                for i in (c_idxs + neg_midxs)
                                    if i >= self.train_metadata.num_entities
                        ])

                    _entity_knn_negs = self.train_knn_index.get_knn_limited_index(
                            c_idxs[1:],
                            include_index_idxs=self.avail_entity_idxs,
                            k=min(num_e_negs, len(self.avail_entity_idxs))
                    )

                    if _entity_knn_negs.shape[1] < num_e_negs:
                        assert _entity_knn_negs.shape[1] == len(self.avail_entity_idxs)
                        _buff = _entity_knn_negs.shape[1] - num_e_negs
                        negs[1:, -num_e_negs:_buff] = _entity_knn_negs
                    else:
                        negs[1:, -num_e_negs:] = _entity_knn_negs

                negatives_list.append(negs)

            else:
                raise NotImplementedError('xdoc neg sampling not implemented yet')
                # produce knn negatives for cluster and append to list

        negs = np.concatenate(negatives_list, axis=0)
        return negs

    def train(self):
        args = self.args

        # set up data structures for choosing available negatives on-the-fly
        if args.clustering_domain == 'within_doc':
            self._neg_choosing_prep()
        else:
            raise NotImplementedError('xdoc not implemented yet')


        global_step = 0
        log_return_dicts = []

        logger.info('Starting training...')

        batch = None
        for epoch in range(args.num_train_epochs):
            logger.info('********** [START] epoch: {} **********'.format(epoch))
            
            num_batches = None
            if get_rank() == 0:
                data_iterator = iter(self.train_dataloader)
                num_batches = len(data_iterator)
            num_batches = broadcast(num_batches, src=0)

            logger.info('num_batches: {}'.format(num_batches))

            for _ in trange(num_batches,
                            desc='Epoch: {} - Batches'.format(epoch),
                            disable=(get_rank() != 0 or args.disable_logging)):
                # get batch from rank0 and broadcast it to the other processes
                if get_rank() == 0:
                    try:
                        next_batch = next(data_iterator)
                        # make sure the cluster_mx is sorted correctly
                        _row, _col, _data = [], [], []
                        current_row = 0
                        ctr = 0
                        for r, d in sorted(zip(next_batch.row, next_batch.data)):
                            if current_row != r:
                                current_row = r
                                ctr = 0
                            _row.append(r)
                            _col.append(ctr)
                            _data.append(d)
                            ctr += 1
                        next_batch = coo_matrix((_data, (_row, _col)), shape=next_batch.shape)
                        negs = self._choose_negs(next_batch)
                        batch = (next_batch, negs)
                    except StopIteration:
                        batch = None

                batch = broadcast(batch, src=0)
                
                if batch is None:
                    break

                # run train_step
                log_return_dicts.append(self.train_step(batch))
                global_step += 1

                # logging stuff for babysitting
                if global_step % args.logging_steps == 0:
                    avg_return_dict = reduce(dict_merge_with, log_return_dicts)
                    for stat_name, stat_value in avg_return_dict.items():
                        logger.info('Average %s: %s at global step: %s',
                                stat_name,
                                str(stat_value/args.logging_steps),
                                str(global_step)
                        )
                        if get_rank() == 0:
                            wandb.log({stat_name : stat_value/args.logging_steps}, step=global_step)
                    log_return_dicts = []

                # refresh the knn index 
                if args.knn_refresh_steps > 0 and global_step % args.knn_refresh_steps == 0:
                    logger.info('Refreshing kNN index...')
                    self.train_knn_index.refresh_index()
                    logger.info('Done.')

            # save the model at the end of every epoch
            if get_rank() == 0:
                self.embed_sub_trainer.save_model(global_step)
            synchronize()

            logger.info('********** [END] epoch: {} **********'.format(epoch))

            # run full evaluation at the end of each epoch
            if args.evaluate_during_training and epoch % 10 == 9:
                if args.do_train_eval:
                    train_eval_metrics = self.evaluate(
                            split='train',
                            suffix='checkpoint-{}'.format(global_step)
                    )
                    if get_rank() == 0:
                        wandb.log(train_eval_metrics, step=global_step)
                if args.do_val:
                    val_metrics = self.evaluate(
                            split='val',
                            suffix='checkpoint-{}'.format(global_step)
                    )
                    if get_rank() == 0:
                        wandb.log(val_metrics, step=global_step)

        logger.info('Training complete')
        #if get_rank() == 0:
        #    embed()
        #synchronize()
        #exit()

    def evaluate(self, split='', suffix=''):
        assert split in ['train', 'val', 'test']
        args = self.args

        logger.info('********** [START] eval: {} **********'.format(split))

        if split == 'train':
            metadata = self.train_metadata
            #knn_index = self.train_knn_index
            example_dir = args.train_cache_dir
        elif split == 'val':
            metadata = self.val_metadata
            #knn_index = self.val_knn_index
            example_dir = args.val_cache_dir
        else:
            metadata = self.test_metadata
            #knn_index = self.test_knn_index
            example_dir = args.test_cache_dir

        ## refresh the knn index
        #knn_index.refresh_index()

        ## save the knn index
        #if get_rank() == 0:
        #    knn_save_fname = os.path.join(args.output_dir, suffix,
        #                                  'knn_index.' + split + '.debug_results.pkl')
        #    with open(knn_save_fname, 'wb') as f:
        #        pickle.dump((knn_index.idxs, knn_index.X), f, pickle.HIGHEST_PROTOCOL)

        embed_metrics = None
        concat_metrics = None
        if args.clustering_domain == 'within_doc':
            #embed_metrics = eval_wdoc(
            #    args, example_dir, metadata, knn_index, self.embed_sub_trainer,
            #    save_fname=os.path.join(args.output_dir, suffix,
            #                            'embed.' + split + '.debug_results.pkl')
            #)
            concat_metrics = eval_wdoc(
                args, example_dir, metadata, self.concat_sub_trainer,
                save_fname=os.path.join(args.output_dir, suffix,
                                        'concat.' + split + '.debug_results.pkl')
            )
            #concat_metrics = {}
        else:
            # FIXME: update after we finalize wdoc changes
            embed_metrics = eval_xdoc(
                args, example_dir, metadata, knn_index, self.embed_sub_trainer
            )
            concat_metrics = eval_xdoc(
                args, example_dir, metadata, knn_index, self.concat_sub_trainer
            )

        embed_metrics = broadcast(embed_metrics, src=0)
        concat_metrics = broadcast(concat_metrics, src=0)

        # pool all of the metrics into one dictionary
        #embed_metrics = {'embed_' + k : v for k, v in embed_metrics.items()}
        concat_metrics = {'concat_' + k : v for k, v in concat_metrics.items()}
        metrics = {}
        #metrics.update(embed_metrics)
        metrics.update(concat_metrics)
        metrics = {split + '_' + k : v for k, v in metrics.items()}

        logger.info(metrics)
        logger.info('********** [END] eval: {} **********'.format(split))
        return metrics
