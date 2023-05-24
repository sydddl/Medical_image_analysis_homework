__author__ = "DeathSprout"

import argparse
import time

import numpy as np
import torch

import util
# ----------------------------
from train_test import Trainer_object
from model import EEGnet

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--epochs', type=int, default=40, help='')
parser.add_argument('--batch_size', type=int, default=4, help='')
parser.add_argument('--num_worker', type=int, default=0, help='进程数,0表示只有主进程')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--learning_rate_step', type=float, default=10, help='learning rate decay step')
parser.add_argument('--gamma', type=float, default=0.1, help='')
parser.add_argument('--save_predictions', type=bool, default=True, help='save model outputs')
parser.add_argument('--save_path', type=str, default="./save/", help='save path')
parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint pth path')
parser.add_argument('--dataset_dir', type=str, default="./data/generate/", help='preprocessed dataset path')
parser.add_argument('--print_every', type=int, default=1000, help='')
parser.add_argument('--class_mod', type=int, default=2, help='2 or 4 (左右手想象/左右手、空、同时左右手')

args = parser.parse_args()


def main():
    print("Using device" + args.device)
    device = torch.device(args.device)
    logger = util.get_logger(args.save_path, __name__, 'info.log', level='INFO')
    logger.info(args)
    Trainer = Trainer_object(args, model = EEGnet())

    if args.checkpoint != None:  # 从断点载入权重
        checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
        Trainer.model.load_state_dict(checkpoint)
        print("load state dict" + args.checkpoint)

    # Training.
    print("start training...", flush=True)
    his_loss = []
    train_time = []
    val_time = []
    for i in range(1, args.epochs + 1):
        train_loss = []
        tt1 = time.time()
        for iter, (x, y) in enumerate(Trainer.trainset):
            x_train = torch.unsqueeze(x.type(torch.float32), 1).to(device)
            y_train = y.type(torch.float32).to(device)
            loss = Trainer.train(x_train, y_train)
            train_loss.append(loss)

            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f} , Lr: {:.6f}'
                logger.info(log.format(iter, train_loss[-1], Trainer.optimizer.param_groups[0]['lr']))
        tt2 = time.time()
        train_time.append(tt2 - tt1)
        Trainer.scheduler.step()

        # Validation.
        valid_loss = []
        vt1 = time.time()
        for iter, (x, y) in enumerate(Trainer.valset):
            x_val = torch.unsqueeze(x.type(torch.float32), 1).to(device)
            y_val = y.type(torch.float32).to(device)
            loss = Trainer.test(x_val, y_val)
            valid_loss.append(loss)
        vt2 = time.time()
        val_time.append(vt2 - vt1)
        mtrain_loss = np.mean(train_loss)
        mvalid_loss = np.mean(valid_loss)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Lr: {:.6f}, Training Time: {:.4f} min/epoch'
        logger.info(log.format(i, mtrain_loss, mvalid_loss, Trainer.optimizer.param_groups[0]['lr'], (tt2 - tt1) / 60))
        torch.save(Trainer.model.state_dict(),
                   args.save_path + "_epoch_" + str(i) + "_" + str(round(mvalid_loss, 4)) + ".pth")
    print("Training finished")
    logger.info("Average Training Time: {:.4f} min/epoch".format(np.mean(train_time) / 60))
    logger.info("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # Testing.
    bestid = np.argmin(his_loss)
    logger.info("The min Valid Loss is :{:.4f}".format(his_loss[bestid]))
    Trainer.model.load_state_dict(
        torch.load(args.save_path + "_epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 4)) + ".pth"))

    outputs = []
    truel = []
    for iter, (x, y) in enumerate(Trainer.testset):
        x_test = torch.unsqueeze(x.type(torch.float32), 1).to(device)
        true_label = y.type(torch.float32).to(device)
        with torch.no_grad():
            preds = Trainer.model(x_test)
        outputs.append(preds.squeeze())
        truel.append(true_label)
    y_out = torch.cat(outputs, dim=0)
    truel = torch.cat(truel, dim=0)
    acc = util.get_acc(y_out, truel)

    print("Training finished")
    logger.info("The best val model test acc is: {:.4f} ".format(round(acc, 4)))

    # Save model outputs.
    if args.save_predictions:
        print('Save outputs in: ', args.save_path)
        np.savez(args.save_path + "predictions_output",
                 predictions=y_out.cpu(),  # Recover original scaling.
                 groundtruth=truel.cpu())


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}s".format(t2 - t1))
