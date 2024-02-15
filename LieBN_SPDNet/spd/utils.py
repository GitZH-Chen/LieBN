import os

import time
import datetime

import random
import geoopt
import numpy as np
import torch as th

from spd.DataLoader.FPHA_Loader import DataLoaderFPHA
from spd.DataLoader.HDM05_Loader import DataLoaderHDM05
from spd.DataLoader.Radar_Loader import DataLoaderRadar

def get_model_name(args):
    if args.model_type == 'SPDNet':
        name = f'{args.seed}-{args.lr}-wd_{args.weight_decay}-{args.model_type}-{args.optimizer}-{args.architecture}-{datetime.datetime.now().strftime("%H_%M")}'
    elif args.model_type == 'SPDNetBN':
        name = f'{args.seed}-{args.lr}-wd_{args.weight_decay}-m_{args.momentum}-{args.model_type}-{args.optimizer}-{args.architecture}-{datetime.datetime.now().strftime("%H_%M")}'
    elif args.model_type in args.total_LieBN_model_types:
        if args.model_type=='SPDNetLieBN_RS':
            model_type = args.model_type + '_init_RS'if args.init_by_RS else args.model_type
        else:
            model_type = args.model_type
        if args.metric == 'AIM' or args.metric == 'LEM':
            name = f'{args.seed}-{args.lr}-wd_{args.weight_decay}-m_{args.momentum}-{model_type}-{args.optimizer}-{args.architecture}-{args.metric}-({args.theta},{args.alpha},{args.beta:.4f})-{datetime.datetime.now().strftime("%H_%M")}'
        elif args.metric== 'LCM':
            name = f'{args.seed}-{args.lr}-wd_{args.weight_decay}-m_{args.momentum}-{model_type}-{args.optimizer}-{args.architecture}-{args.metric}-({args.theta})-{datetime.datetime.now().strftime("%H_%M")}'
    else:
        raise Exception('unknown metric {} or model'.format(args.metric,args.model_type))
    return name

def get_dataset_settings(args):
    if args.dataset=='FPHA':
        class_num = 45
        DataLoader = DataLoaderFPHA(args.path,args.batchsize)
    elif args.dataset=='HDM05':
        class_num = 117
        pval = 0.5
        DataLoader = DataLoaderHDM05(args.path, pval, args.batchsize)
    elif args.dataset== 'RADAR' :
        class_num = 3
        pval = 0.25
        DataLoader = DataLoaderRadar(args.path,pval,args.batchsize)
    else:
        raise Exception('unknown dataset {}'.format(args.dataset))
    return class_num,DataLoader

def set_seed_thread(seed,threadnum):
    th.set_num_threads(threadnum)
    seed = seed
    random.seed(seed)
    # th.cuda.set_device(args.gpu)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)

def optimzer(parameters,lr,mode='AMSGRAD',weight_decay=0.):
    if mode=='ADAM':
        optim = geoopt.optim.RiemannianAdam(parameters, lr=lr,weight_decay=weight_decay)
    elif mode=='SGD':
        optim = geoopt.optim.RiemannianSGD(parameters, lr=lr,weight_decay=weight_decay)
    elif mode=='AMSGRAD':
        optim = geoopt.optim.RiemannianAdam(parameters, lr=lr,amsgrad=True,weight_decay=weight_decay)
    else:
        raise Exception('unknown optimizer {}'.format(mode))
    return optim

def training_loop(model, data_loader, opti, loss_fn,writer, args,model_path, begin_epoch):
    acc_val = [];loss_val = [];acc_train = [];loss_train = []
    # training loop
    for epoch in range(begin_epoch, args.epochs):
        # train one epoch
        start = time.time()
        temp_loss_train, temp_acc_train = [], []
        model.train()
        for local_batch, local_labels in data_loader._train_generator:
            opti.zero_grad()
            out = model(local_batch)
            l = loss_fn(out, local_labels)
            acc, loss = (out.argmax(1) == local_labels).cpu().numpy().sum() / out.shape[0], l.cpu().data.numpy()
            temp_loss_train.append(loss)
            temp_acc_train.append(acc)
            l.backward()
            opti.step()
        if args.is_gpu:
            th.cuda.synchronize()
        end = time.time()
        acc_train.append(np.asarray(temp_acc_train).mean() * 100)
        loss_train.append(np.asarray(temp_loss_train).mean())

        # validation
        acc_val_list = [];loss_val_list = [];y_true, y_pred = [], []
        model.eval()
        with th.no_grad():
            for local_batch, local_labels in data_loader._test_generator:
                out = model(local_batch)
                l = loss_fn(out, local_labels)
                predicted_labels = out.argmax(1)
                y_true.extend(list(local_labels.cpu().numpy()));
                y_pred.extend(list(predicted_labels.cpu().numpy()))
                acc, loss = (predicted_labels == local_labels).cpu().numpy().sum() / out.shape[0], l.cpu().data.numpy()
                acc_val_list.append(acc)
                loss_val_list.append(loss)
        loss_val.append(np.asarray(loss_val_list).mean())
        acc_val.append(np.asarray(acc_val_list).mean() * 100)
        if args.is_writer:
            writer.add_scalar('Loss/val', loss_val[epoch], epoch)
            writer.add_scalar('Accuracy/val', acc_val[epoch], epoch)
            writer.add_scalar('Loss/train', loss_train[epoch], epoch)
            writer.add_scalar('Accuracy/train', acc_train[epoch], epoch)
        print(
            '{}: time: {:.2f}, Val acc: {:.2f}, loss: {:.2f}, at epoch {:d}/{:d} '.format(
                args.modelname,end - start, acc_val[epoch], loss_val[epoch], epoch + 1, args.epochs))
        if epoch + 1 == args.epochs and args.is_save:
            th.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'lr': args.lr,
                'acc_val': acc_val,
                'acc_train': acc_train,
                'loss_val': loss_val,
                'loss_train': loss_train
            }, model_path + '-' + str(epoch))
    print('{}: Final validation accuracy: {}%'.format(args.modelname,acc_val[-1]))
    if args.is_writer:
        writer.close()
    return acc_val

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

def resuming_writer(begin_epoch, writer,loss_val,loss_train,acc_val,acc_train):
    for epoch in range(begin_epoch):
        writer.add_scalar('Loss/val', loss_val[epoch], epoch)
        writer.add_scalar('Accuracy/val', acc_val[epoch], epoch)
        writer.add_scalar('Loss/train', loss_train[epoch], epoch)
        writer.add_scalar('Accuracy/train', acc_train[epoch], epoch)

def parse_cfg(args,cfg):
    # setting args from cfg

    args.seed = cfg.fit.seed
    args.model_type = cfg.nnet.model.model_type
    args.is_save = cfg.fit.is_save

    if args.model_type in args.total_BN_model_types:
        args.BN_type = cfg.nnet.model.BN_type
        args.momentum = cfg.nnet.model.momentum
        if args.model_type in args.total_LieBN_model_types:
            args.metric = cfg.nnet.model.metric
            args.theta = cfg.nnet.model.theta
            args.alpha = cfg.nnet.model.alpha
            args.beta = eval(cfg.nnet.model.beta) if isinstance(cfg.nnet.model.beta, str) else cfg.nnet.model.beta

    args.dataset = cfg.dataset.name
    args.architecture = cfg.dataset.architecture
    args.path = cfg.dataset.path

    args.optimizer = cfg.nnet.optimizer.mode
    args.lr = cfg.nnet.optimizer.lr
    args.weight_decay = cfg.nnet.optimizer.weight_decay

    args.epochs = cfg.fit.epochs
    args.batchsize = cfg.fit.batch_size

    args.threadnum = cfg.fit.threadnum
    args.is_writer = cfg.fit.is_writer
    args.cycle = cfg.fit.cycle

    # get model name
    args.modelname = get_model_name(args)

    return args