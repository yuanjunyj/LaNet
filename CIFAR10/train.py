# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import  os
import  sys
import  time
from datetime import timedelta
import  glob
import  numpy as np
import  torch
import  utils
import  logging
import  argparse
import  torch.nn as nn
import  torch.utils
import  torchvision.datasets as dset
import  torch.backends.cudnn as cudnn
from nasnet_set import *
from collections import namedtuple
import hashlib
from datetime import datetime
from model import NetworkCIFAR as Network
from utils import ModelEma
from utils import *
from cutmix.cutmix import CutMix
from cutmix.utils import CutMixCrossEntropyLoss
from sklearn.model_selection._split import StratifiedShuffleSplit
from torch.utils.data.dataset import Subset




parser = argparse.ArgumentParser("cifar10")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--lr', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--wd', type=float, default=3e-4, help='weight decay')
parser.add_argument('--no_bias_decay', action='store_true', default=False, help='prevent bias decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_ch', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--mixup', action='store_true', default=False, help='use mixup')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--drop_path_prob_decay', action='store_true', default=False, help='use drop_path_prob decay')
parser.add_argument('--smoothing', type=float, default=0.0, help='label smoothing')
parser.add_argument('--exp_path', type=str, default='exp/cifar10', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')





parser.add_argument('--model_ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model_ema_force_cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

parser.add_argument('--track_ema', action='store_true', default=False, help='track ema')
parser.add_argument('--auto_augment', action='store_true', default=False)
parser.add_argument('--rand_augment', action='store_true', default=False)


parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--port", type=int, default=9000)

parser.add_argument("--name", type=str, default="")
parser.add_argument('--finetune', type=int, default=0, help='Finetune')


args = parser.parse_args()

if args.name == "":
    args.name = args.arch

# checkpoints/
args.save = '../OUTPUT/auto_aug-{}-{}-{}-{}-{}-{}-{}-{}'.format(args.epochs, args.name, args.init_ch, args.model_ema,
                                                         args.model_ema_decay, args.drop_path_prob,
                                                               args.layers, args.cutout_length)


print("save path:", args.save)


continue_train = False
if os.path.exists(args.save+'/model.pt'):
    continue_train = True

if not continue_train:
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
if args.finetune == 0:
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
else:
    fh = logging.FileHandler(os.path.join(args.save, 'log_finetune' + str(args.finetune) + '.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)



def flatten_params(model):
    return torch.sum( torch.cat([param.data.view(-1) for param in model.parameters()], 0))

def init_distributed(params):
    params.num_gpus = int(os.environ["WORLD_SIZE"]) \
        if "WORLD_SIZE" in os.environ else 1
    # params.world_size = torch.distributed.get_world_size()
    params.distributed = params.num_gpus > 1

    if params.distributed and params.local_rank > -1:
        print("=> init process group start")
        torch.cuda.set_device(params.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
            timeout=timedelta(minutes=180))
        print("=> init process group end")

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    torch.distributed.barrier()

class MeterReduce:
    def __init__(self, local_rank):
        self.local_rank = local_rank

    def reduce(self, meter, size):
        rank = self.local_rank
        meter_sum = torch.FloatTensor([meter * size]).cuda(rank)
        meter_count = torch.FloatTensor([size]).cuda(rank)
        torch.distributed.reduce(meter_sum, 0)
        torch.distributed.reduce(meter_count, 0)
        meter_avg = meter_sum / meter_count

        return meter_avg.item(), meter_count.item()

class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        # indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples

def main():

    print("device_count", torch.cuda.device_count())

    init_distributed(args)
    
    # torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.enabled = True
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    cur_epoch = 0

    # Mask Orignal Def
    net = eval(args.arch)
    print(net)
    code = gen_code_from_list(net, node_num=int((len(net) / 4)))
    genotype = translator([code, code], max_node=int((len(net) / 4)))

    # Arch from Github Issue
    # genotype = Genotype(
    #     normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2),
    #             ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_5x5', 0)], normal_concat=[2, 3, 4, 5],
    #     reduce=[('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 2),
    #             ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=[2, 3, 4, 5])
    # print(genotype)

    # Mask from Nasbench301
    # genotype = mask2config_nasbench301(net)
    # print(genotype)

    # Mask from NASNet Supernet
    # with open("net_config.json", 'r') as config_file:
    #     net = json.loads( config_file.read() )["mask"]
    # genotype = mask2config_nasnet(net)
    # logging.info("MASK = ", net)
    # logging.info("GENOTYPE = %s", genotype)

    model_ema = None

    if not continue_train:

        print('train from the scratch')
        model = Network(args.init_ch, 10, args.layers, args.auxiliary, genotype)
        if args.distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # .cuda()
        model.to(torch.device('cuda'))

        print("model init params values:", flatten_params(model))

        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

        criterion = CutMixCrossEntropyLoss(True).cuda()

        if args.model_ema:
            model_ema = ModelEma(
                model,
                decay=args.model_ema_decay,
                device='cpu' if args.model_ema_force_cpu else '')
        if args.no_bias_decay:
            optimizer = torch.optim.SGD([
                    {'params': (p for name, p in model.named_parameters() if 'bias' not in name)},
                    {'params': (p for name, p in model.named_parameters() if 'bias' in name), 'weight_decay': 0.}
                ],
                args.lr,
                momentum=args.momentum,
                weight_decay=args.wd
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(),
                args.lr,
                momentum=args.momentum,
                weight_decay=args.wd
            )
                
    else:
        print('continue train from checkpoint')

        model = Network(args.init_ch, 10, args.layers, args.auxiliary, genotype)
        if args.distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        #.cuda()

        ckpt_path = args.save+'/model.pt'
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        model.to(torch.device('cuda'))
        model.load_state_dict(checkpoint['model_state_dict'])
        cur_epoch = checkpoint['epoch']

        criterion = CutMixCrossEntropyLoss(True).cuda()

        if args.no_bias_decay:
            optimizer = torch.optim.SGD(
                [
                    {'params': (p for name, p in model.named_parameters() if 'bias' not in name)},
                    {'params': (p for name, p in model.named_parameters() if 'bias' in name), 'weight_decay': 0.}
                ],
                args.lr,
                momentum=args.momentum,
                weight_decay=args.wd
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(),
                args.lr,
                momentum=args.momentum,
                weight_decay=args.wd
            )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

        if args.model_ema:

            model_ema = ModelEma(
                model,
                decay=args.model_ema_decay,
                device='cpu' if args.model_ema_force_cpu else '',
                resume=ckpt_path)


    if args.local_rank > -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank
        )

    if args.finetune > 0:
        args.epochs += args.finetune
        args.drop_path_prob = 0.25

    train_transform, valid_transform = utils._auto_data_transforms_cifar10(args)

    ds_train = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    # args.cv = -1
    # if args.cv >= 0:
    #     sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    #     sss = sss.split(list(range(len(ds_train))), ds_train.targets)
    #     for _ in range(args.cv + 1):
    #         train_idx, valid_idx = next(sss)
    #     ds_valid = Subset(ds_train, valid_idx)
    #     ds_train = Subset(ds_train, train_idx)
    # else:
    #     ds_valid = Subset(ds_train, [])

    if args.local_rank > -1:
        # distributed
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            ds_train, shuffle=True
        )
        train_queue = torch.utils.data.DataLoader(
            CutMix(ds_train, 10,
                beta=1.0, prob=0.5, num_mix=2, use_mixup=args.mixup, smoothing=args.smoothing),
            batch_size=args.batch_size, sampler=train_sampler, num_workers=0, pin_memory=True)

        plain_train_sampler = SequentialDistributedSampler(ds_train, batch_size=args.batch_size)
        plain_train_queue = torch.utils.data.DataLoader(
            ds_train,
            batch_size=args.batch_size * 2, sampler=train_sampler, num_workers=2, pin_memory=True)

        ds_valid = dset.CIFAR10(root=args.data, train=False, transform=valid_transform)
        valid_sampler = SequentialDistributedSampler(ds_valid, batch_size=args.batch_size)
        valid_queue = torch.utils.data.DataLoader(
            ds_valid,
            batch_size=args.batch_size * 2, sampler=valid_sampler, num_workers=2, pin_memory=True)
    else:
        train_queue = torch.utils.data.DataLoader(
            CutMix(ds_train, 10,
                beta=1.0, prob=0.5, num_mix=2, use_mixup=args.mixup, smoothing=args.smoothing),
            batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        
        plain_train_queue = torch.utils.data.DataLoader(
            ds_train,
            batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        valid_queue = torch.utils.data.DataLoader(
            dset.CIFAR10(root=args.data, train=False, transform=valid_transform),
            batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    best_acc = 0.0
    best_acc_ema = 0.0

    if continue_train:
        for i in range(cur_epoch+1):
            scheduler.step()

    for epoch in range(cur_epoch, args.epochs):
        if args.local_rank == -1 or args.local_rank == 0:
            print('cur_epoch is', epoch)

        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        
        if args.local_rank > -1:
            model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        else:
            model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        if model_ema is not None:
            model_ema.ema.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        if args.distributed and hasattr(train_queue.sampler, 'set_epoch'):
            train_queue.sampler.set_epoch(epoch)

        # Training
        if args.distributed:
            train_acc, loss, cnt = train(train_queue, model, criterion, optimizer, epoch, model_ema)
            # logging.info("train_loss_raw {} {} {}".format(args.local_rank, loss, cnt))
            synchronize()
            reducer = MeterReduce(args.local_rank)
            loss, cnt = reducer.reduce(loss, cnt)
            # logging.info("train_loss {} {} {}".format(args.local_rank, loss, cnt))
            synchronize()

            # logging.info("train_acc_raw {} {} {}".format(args.local_rank, train_acc, cnt))
            synchronize()
            reducer = MeterReduce(args.local_rank)
            train_acc, cnt = reducer.reduce(train_acc, cnt)
            # logging.info("train_acc {} {} {}".format(args.local_rank, train_acc, cnt))
            synchronize()

            if args.local_rank == 0:
                logging.info("train_acc_sync: {}".format(train_acc))
                logging.info("train_loss_sync: {}".format(loss))
        else:
            train_acc, loss, cnt = train(train_queue, model, criterion, optimizer, epoch, model_ema)
            logging.info("train_acc {}".format(train_acc))
            logging.info("loss {}".format(loss))

        # Infer Training Set
        # if args.distributed:
        #     plain_train_acc, plain_train_obj, cnt = infer(plain_train_queue, model, criterion)
        #     # logging.info("plain_train_acc_raw {} {} {}".format(args.local_rank, plain_train_acc, cnt))
        #     synchronize()
        #     reducer = MeterReduce(args.local_rank)
        #     plain_train_acc, cnt = reducer.reduce(plain_train_acc, cnt)
        #     # logging.info("plain_train_acc {} {} {}".format(args.local_rank, plain_train_acc, cnt))
        #     synchronize()

        #     # logging.info("plain_train_loss_raw {} {} {}".format(args.local_rank, plain_train_obj, cnt))
        #     synchronize()
        #     reducer = MeterReduce(args.local_rank)
        #     plain_train_obj, cnt = reducer.reduce(plain_train_obj, cnt)
        #     # logging.info("plain_train_loss {} {} {}".format(args.local_rank, plain_train_obj, cnt))
        #     synchronize()

        #     if args.local_rank == 0:
        #         logging.info('plain_train_acc_sync: %f', plain_train_acc)
        #         logging.info('plain_train_loss_sync: %f', plain_train_obj)
        # else:
        #     plain_train_acc, plain_train_obj, cnt = infer(plain_train_queue, model, criterion)
        #     logging.info('plain_train_acc: %f', plain_train_acc)
        #     logging.info('plain_train_loss: %f', plain_train_obj)

        # if model_ema is not None and not args.model_ema_force_cpu:
        #     if args.distributed:
        #         plain_train_acc_ema, plain_train_obj_ema, cnt_ema = infer(plain_train_queue, model_ema.ema, criterion, ema=True)
        #         # logging.info("plain_train_acc_ema_raw {} {} {}".format(args.local_rank, plain_train_acc_ema, cnt_ema))
        #         synchronize()
        #         reducer = MeterReduce(args.local_rank)
        #         plain_train_acc_ema, cnt_ema = reducer.reduce(plain_train_acc_ema, cnt_ema)
        #         # logging.info("plain_train_acc_ema {} {} {}".format(args.local_rank, plain_train_acc_ema, cnt_ema))
        #         synchronize()

        #         # logging.info("plain_train_loss_ema_raw {} {} {}".format(args.local_rank, plain_train_obj_ema, cnt_ema))
        #         synchronize()
        #         reducer = MeterReduce(args.local_rank)
        #         plain_train_obj_ema, cnt_ema = reducer.reduce(plain_train_obj_ema, cnt_ema)
        #         # logging.info("plain_train_loss_ema {} {} {}".format(args.local_rank, plain_train_obj_ema, cnt_ema))
        #         synchronize()

        #         if args.local_rank == 0:
        #             logging.info('plain_train_acc_ema_sync: %f', plain_train_acc_ema)
        #             logging.info('plain_train_loss_ema_sync: %f', plain_train_obj_ema)
        #     else:
        #         plain_train_acc_ema, plain_train_obj_ema, cnt_ema = infer(plain_train_queue, model_ema.ema, criterion, ema=True)
        #         logging.info('plain_train_acc_ema %f', plain_train_acc_ema)
        #         logging.info('plain_train_loss_ema %f', plain_train_obj_ema)
        
        # Infer Test Set
        if epoch % 1 == 0:
            if args.distributed:
                valid_acc, valid_obj, cnt = infer(valid_queue, model, criterion)
                # logging.info("valid_acc_raw {} {} {}".format(args.local_rank, valid_acc, cnt))
                synchronize()
                reducer = MeterReduce(args.local_rank)
                valid_acc, cnt = reducer.reduce(valid_acc, cnt)
                # logging.info("valid_acc {} {} {}".format(args.local_rank, valid_acc, cnt))
                synchronize()

                # logging.info("valid_loss_raw {} {} {}".format(args.local_rank, valid_obj, cnt))
                synchronize()
                reducer = MeterReduce(args.local_rank)
                valid_obj, cnt = reducer.reduce(valid_obj, cnt)
                # logging.info("valid_loss {} {} {}".format(args.local_rank, valid_obj, cnt))
                synchronize()

                if args.local_rank == 0:
                    logging.info('valid_acc_sync: %f', valid_acc)
                    logging.info('valid_loss_sync: %f', valid_obj)
            else:
                valid_acc, valid_obj, cnt = infer(valid_queue, model, criterion)
                logging.info('valid_acc: %f', valid_acc)
                logging.info('valid_loss: %f', valid_obj)

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed:
                    valid_acc_ema, valid_obj_ema, cnt_ema = infer(valid_queue, model_ema.ema, criterion, ema=True)
                    # logging.info("valid_acc_ema_raw {} {} {}".format(args.local_rank, valid_acc_ema, cnt_ema))
                    synchronize()
                    reducer = MeterReduce(args.local_rank)
                    valid_acc_ema, cnt_ema = reducer.reduce(valid_acc_ema, cnt_ema)
                    # logging.info("valid_acc_ema {} {} {}".format(args.local_rank, valid_acc_ema, cnt_ema))
                    synchronize()

                    # logging.info("valid_loss_ema_raw {} {} {}".format(args.local_rank, valid_obj_ema, cnt_ema))
                    synchronize()
                    reducer = MeterReduce(args.local_rank)
                    valid_obj_ema, cnt_ema = reducer.reduce(valid_obj_ema, cnt_ema)
                    # logging.info("valid_loss_ema {} {} {}".format(args.local_rank, valid_obj_ema, cnt_ema))
                    synchronize()

                    if args.local_rank == 0:
                        logging.info('valid_acc_ema_sync: %f', valid_acc_ema)
                        logging.info('valid_loss_ema_sync: %f', valid_obj_ema)
                else:
                    valid_acc_ema, valid_obj_ema, cnt_ema = infer(valid_queue, model_ema.ema, criterion, ema=True)
                    logging.info('valid_acc_ema %f', valid_acc_ema)
                    logging.info('valid_loss_ema %f', valid_obj_ema)
                

        if (args.distributed and args.local_rank == 0) or not args.distributed:
            if valid_acc > best_acc:
                best_acc = valid_acc
                if not args.distributed:
                    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(args.save, 'top1.pt'))
                else:
                    torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(args.save, 'top1.pt'))

            if valid_acc_ema > best_acc_ema:
                best_acc_ema = valid_acc_ema
                if not args.distributed:
                    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(args.save, 'top1.pt'))
                else:
                    torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(args.save, 'top1.pt'))

            if model_ema is not None:
                if not args.distributed:
                    torch.save(
                        {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                        'state_dict_ema': get_state_dict(model_ema)},
                        os.path.join(args.save, 'model.pt'))
                else:
                    torch.save(
                        {'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                        'state_dict_ema': get_state_dict(model_ema)},
                        os.path.join(args.save, 'model.pt'))
            else:
                if not args.distributed:
                    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(args.save, 'model.pt'))
                else:
                    torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(args.save, 'model.pt'))

            logging.info('best_acc: %f', best_acc)
            logging.info('best_acc_ema: %f', best_acc_ema)


def train(train_queue, model, criterion, optimizer, epoch, model_ema=None):

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    model.train()

    for step, (x, target) in enumerate(train_queue):
        x = x.cuda()
        target = target.cuda(non_blocking=True)
        optimizer.zero_grad()
        logits, logits_aux = model(x, args.drop_path_prob_decay)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux
        
        if len(target.size()) == 1:
            # print(target.size())
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

        # prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        losses.update(loss.item(), x.size(0))
        # top1.update(prec1.item(), x.size(0))
        # top5.update(prec5.item(), x.size(0))

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()

        # if step % args.report_freq == 0:
        #     logging.info('train %03d', step)

    return top1.avg, losses.avg, losses.cnt


def infer(valid_queue, model, criterion, ema=False):

    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()

    for step, (x, target) in enumerate(valid_queue):
        x = x.cuda()
        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            logits, _ = model(x, args.drop_path_prob_decay)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = x.size(0)

            torch.cuda.synchronize()

            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

        # if step % args.report_freq == 0:
        #     if not ema:
        #         logging.info('>>Validation: %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
        #     else:
        #         logging.info('>>Validation_ema: %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg, objs.cnt

if __name__ == '__main__':
    main()
