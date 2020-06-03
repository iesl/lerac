import os
import sys
import random
import logging
import pickle
import time
from datetime import datetime, timedelta
import numpy as np
import torch
import torch.distributed as dist

from IPython import embed


START_HGHLGHT_TOKEN = '[unused1]'
END_HGHLGHT_TOKEN = '[unused2]'


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def all_same(lst):
    return lst.count(lst[0]) == len(lst)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def setup_cuda_and_distributed(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend='nccl')
        args.n_gpu = 1
        args.world_size = dist.get_world_size()
    args.device = device


class DistributedCache(object):
    ''' Simple caching object '''

    def __init__(self, args):
        self.data = {}
        self.local_rank = args.local_rank

    def get(self, keys):
        '''
        param:keys is a list of keys
        '''
        return [self.data.get(k) for k in keys]

    def set(self, keys, data):
        '''
        param:keys is a list of keys
        param:data is a list of pytorch tensors with the same length as keys
        '''
        for k, d in zip(keys, data):
            self.data[k] = d

    def sync(self):
        '''
        sync data across processes
        '''


def dict_merge_with(d1, d2, fn=lambda x, y: x + y):
    res = d1.copy() # "= dict(d1)" for lists of tuples
    for key, val in d2.items(): # ".. in d2" for lists of tuples
        try:
            res[key] = fn(res[key], val)
        except KeyError:
            res[key] = val
    return res


def initialize_exp(args, logger_filename='train.log'):
    """
    Initialize the experiment:
    - dump parameters
    - create a logger
    - set the random seed
    - setup distributed computation
    """
    # setup cuda using torch's distributed framework
    setup_cuda_and_distributed(args)

    # random seed
    set_seed(args)

    # don't overwrite previous output directory
    if (os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir):
        raise ValueError("Output directory ({}) already exists and is not "
                         "empty. Use --overwrite_output_dir "
                         "to overcome.".format(args.output_dir))

    # create output directory and dump parameters
    if args.local_rank in [-1, 0]:
        # create output directory
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        # args file prefix
        if args.do_train:
            prefix = "train"
        elif args.do_train_eval:
            prefix = "train_eval"
        elif args.do_val:
            prefix = "val"
        elif args.do_test:
            prefix = "test"
        else:
            raise ValueError("No valid train or validation mode selected")
        args_file = prefix + "_args.pkl"
        pickle.dump(args, open(os.path.join(args.output_dir, args_file), "wb"))
    dist.barrier()

    # get running command
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith('--'):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            command.append("'%s'" % x)
    command = ' '.join(command)
    args.command = command 


    # create a logger
    logger = create_logger(args, logger_filename)
    logger.info('============ Initialized logger ============')
    logger.info('\n'.join(['%s: %s' % (k, str(v))
                           for k, v in sorted(dict(vars(args)).items())]))
    logger.info('The experiment will be stored in %s\n' % args.output_dir)
    logger.info('Running command: %s\n' % args.command)
    return logger


class LogFormatter():

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''


def create_logger(args, logger_filename):
    """
    Create a logger.
    """
    assert logger_filename is not None

    # create log formatter
    log_formatter = LogFormatter()

    # setup multiple filepaths for logger
    filepath_exp = os.path.join(args.output_dir, logger_filename)
    filename_prefix = ".".join(logger_filename.split(".")[:-1])
    logger_filename = ".".join([
            args.task_name,
            os.path.basename(os.path.dirname(args.data_dir)),
            filename_prefix,
            datetime.now().strftime("%d_%m_%Y-%H:%M:%S"),
            "log"
    ])
    filepath_log = os.path.join(args.log_dir, logger_filename)

    # create file handlers and set level to debug
    file_handlers = []
    for filepath in [filepath_exp, filepath_log]:
        fh = logging.FileHandler(filepath, "a")
        fh.setLevel(logging.DEBUG if args.local_rank in [0, -1]
                              else logging.WARN)
        fh.setFormatter(log_formatter)
        file_handlers.append(fh)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO if args.local_rank in [0, -1]
                             else logging.WARN)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG if args.local_rank in [0, -1]
                    else logging.WARN)
    logger.propagate = False
    for fh in file_handlers:
        logger.addHandler(fh)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    return logger
