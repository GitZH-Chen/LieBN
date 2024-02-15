from spd.utils import get_dataset_settings
from Network import Get_Model
import spd.utils as nn_spd_utils

from torch.utils.tensorboard import SummaryWriter

import os
import datetime
import time
import logging

import torch as th
import torch.nn as nn
import numpy as np
import fcntl

def training(cfg,args):
    args=nn_spd_utils.parse_cfg(args,cfg)

    #set logger
    logger = logging.getLogger(args.modelname)
    logger.setLevel(logging.INFO)
    args.logger = logger
    logger.info('begin model {} on dataset: {}'.format(args.modelname,args.dataset))

    #set seed and threadnum
    nn_spd_utils.set_seed_thread(args.seed,args.threadnum)

    # set dataset, model and optimizer
    args.class_num, args.DataLoader = get_dataset_settings(args)
    model = Get_Model.get_model(args)
    loss_fn = nn.CrossEntropyLoss()
    args.loss_fn = loss_fn.cuda()
    args.opti = nn_spd_utils.optimzer(model.parameters(), lr=args.lr, mode=args.optimizer,weight_decay=args.weight_decay)
    # begin training
    val_acc = training_loop(model,args)

    return val_acc

def training_loop(model, args):
    #setting tensorboard
    if args.is_writer:
        args.writer_path = os.path.join('./tensorboard_logs/',f"{args.modelname}")
        args.logger.info('writer path {}'.format(args.writer_path))
        args.writer = SummaryWriter(args.writer_path)

    acc_val = [];loss_val = [];acc_train = [];loss_train = [];training_time=[]
    logger = args.logger
    # training loop
    for epoch in range(0, args.epochs):
        # train one epoch
        start = time.time()
        temp_loss_train, temp_acc_train = [], []
        model.train()
        for local_batch, local_labels in args.DataLoader._train_generator:
            args.opti.zero_grad()
            out = model(local_batch)
            l = args.loss_fn(out, local_labels)
            acc, loss = (out.argmax(1) == local_labels).cpu().numpy().sum() / out.shape[0], l.cpu().data.numpy()
            temp_loss_train.append(loss)
            temp_acc_train.append(acc)
            l.backward()
            args.opti.step()
        end = time.time()
        training_time.append(end-start)
        acc_train.append(np.asarray(temp_acc_train).mean() * 100)
        loss_train.append(np.asarray(temp_loss_train).mean())

        # validation
        acc_val_list = [];loss_val_list = [];y_true, y_pred = [], []
        model.eval()
        with th.no_grad():
            for local_batch, local_labels in args.DataLoader._test_generator:
                out = model(local_batch)
                l = args.loss_fn(out, local_labels)
                predicted_labels = out.argmax(1)
                y_true.extend(list(local_labels.cpu().numpy()));
                y_pred.extend(list(predicted_labels.cpu().numpy()))
                acc, loss = (predicted_labels == local_labels).cpu().numpy().sum() / out.shape[0], l.cpu().data.numpy()
                acc_val_list.append(acc)
                loss_val_list.append(loss)
        loss_val.append(np.asarray(loss_val_list).mean())
        acc_val.append(np.asarray(acc_val_list).mean() * 100)

        if args.is_writer:
            args.writer.add_scalar('Loss/val', loss_val[epoch], epoch)
            args.writer.add_scalar('Accuracy/val', acc_val[epoch], epoch)
            args.writer.add_scalar('Loss/train', loss_train[epoch], epoch)
            args.writer.add_scalar('Accuracy/train', acc_train[epoch], epoch)

        if epoch % args.cycle == 0:
                logger.info(
                    'Time: {:.2f}, Val acc: {:.2f}, loss: {:.2f} at epoch {:d}/{:d} '.format(
                        training_time[epoch], acc_val[epoch], loss_val[epoch], epoch + 1, args.epochs))

    if args.is_save:
        average_time = np.asarray(training_time[-10:]).mean()
        final_val_acc = acc_val[-1]
        final_results = f'Final validation accuracy : {final_val_acc:.2f}% with average time: {average_time:.2f}'
        final_results_path = os.path.join(os.getcwd(), 'final_results_' + args.dataset)
        logger.info(f"results file path: {final_results_path}, and saving the results")
        write_final_results(final_results_path, args.modelname + '- ' + final_results)
        torch_results_dir = './torch_resutls'
        if not os.path.exists(torch_results_dir):
            os.makedirs(torch_results_dir)
        th.save({
            'acc_val': acc_val,
        }, os.path.join(torch_results_dir,args.modelname.rsplit('-',1)[0]))

    if args.is_writer:
        args.writer.close()
    return acc_val

def write_final_results(file_path,message):
    # Create a file lock
    with open(file_path, "a") as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)  # Acquire an exclusive lock

        # Write the message to the file
        file.write(message + "\n")

        fcntl.flock(file.fileno(), fcntl.LOCK_UN)