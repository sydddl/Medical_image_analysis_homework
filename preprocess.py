__author__ = "DeathSprout"

import os
import mne
import random
from glob import glob
import numpy as np
import argparse


def preprocess_data(data, f_low=2, f_high=40, notch=False, bandpass_filter=False):
    if notch:
        data.load_data().notch_filter(
            freqs=(60), picks=mne.pick_types(data.info, eeg=True),
            method="spectrum_fit", filter_length="30s", verbose="warning"
        )
    if bandpass_filter:
        data.load_data().filter(f_low, f_high, method="fir", verbose="warning")  # 带通滤波

    return data


class generate_preprocess_dataset():
    '''pytho
    Generate train, validation and test data samples from 103 subject EEG session timecourses.
    '''

    def __init__(self, path, output_dir, class_mod=2, samples=480, divide=[7, 1, 2]):

        '''
        class_mod: 分类模式
        samples : 单个样本EGG的采样点数量
        divide :  数据集划分比例
        '''

        print("loaded dataset from: " + path)
        self.mod = class_mod
        SUBS = glob(path + 'S[0-9]*')
        FNAMES = sorted([x[-4:] for x in SUBS])
        FNAMES = FNAMES[:109]
        # 移除几个标注错误的个体（可见数据集原文）
        bad_sub = ['S038', 'S088', 'S089', 'S092', 'S100', 'S104']
        try:
            for sub in bad_sub:
                FNAMES.remove(sub)
        except:
            pass

        # 要做 4类 [睁眼静息、左手想象、右手想象、双手想象(label是左右手的结合)]，闭眼和双脚分到其他，具体见label_to_int
        self.imagined = '04,08,12'.split(',')
        self.imagined_both = '06,10,14'.split(',')
        if self.mod == 4:
            file_numbers = self.imagined + self.imagined_both + ['02']
        elif self.mod == 2:
            file_numbers = self.imagined
        else:
            raise Exception("Invalid class_mod value %d" % class_mod)

        X = []
        Y = []

        for subj in FNAMES:
            # Load the file names for given subject
            fnames = glob(os.path.join(path, subj, subj + 'R*.edf'))
            fnames = [name for name in fnames if name[-6:-4] in file_numbers]
            # Iterate over the trials for each subject
            for file_name in fnames:
                self.file = file_name[-6:-4]
                raw = mne.io.read_raw_edf(file_name, verbose="warning")  # 读取数据，并不让mne产生非报错的日志
                events_from_annot, event_dict = mne.events_from_annotations(raw, verbose="warning")
                raw_pre = preprocess_data(raw, f_low= args.lowpass, f_high= args.highpass,
                                          notch=True, bandpass_filter=True)[:][0]

                if self.mod == 2:  # 103x3x15=4635
                    for events in events_from_annot:
                        if events[2] > 1:  # 三个文件加起来T1数量比T2的trial多一个,也没啥影响
                            begin = events[0]
                            x_data = raw_pre[:, begin:begin + samples]  # 默认三秒(480)采样
                            # (++,64,480)
                            X.append(x_data)
                            y_label = self.label_to_int(events[2])
                            Y.append(y_label)
                        else:
                            pass
            print("loading subject: " + subj)

        # 随机打乱数据集
        xyzip = list(zip(X, Y))
        random.shuffle(xyzip)  # 作为元组一起打乱
        X, Y = zip(*xyzip)
        X, Y = np.array(X), np.array(Y)
        num = X.shape[0]
        divide = np.array(divide)
        train_num = int((divide[0] / 10) * num)
        val_num = int((divide[1] / 10) * num)

        x_train, y_train = X[:train_num], Y[:train_num]
        x_val, y_val = X[train_num:train_num + val_num], Y[train_num:train_num + val_num]
        x_test, y_test = X[train_num + val_num:], Y[train_num + val_num:]

        if output_dir is not None:
            print('### SAVE DATA ###')
            print('Save in: ' + output_dir)
            for cat in ["train", "val", "test"]:
                _x, _y = locals()["x_" + cat], locals()["y_" + cat]
                print(cat, "x: ", _x.shape, "y:", _y.shape)
                np.savez_compressed(
                    os.path.join(output_dir, "%s.npz" % cat),
                    x=_x,
                    y=_y,
                )
        print('Done.')

    # 标签转换
    def label_to_int(self, label):

        if self.file in self.imagined:
            arg = True
        else:
            arg = False

        if self.mod == 4:
            if label == 1 and arg is True:  # corresponds to rest
                return [0, 0, 0]
            elif label == 2 and arg is True:  # the left fist
                return [1, 0, 0]
            elif label == 3:  # the right fist
                return [0, 1, 0]
            elif label == 2 and arg is False:  # both fists
                return [1, 1, 0]
            else:  # both feet or Baseline, eyes closed
                return [0, 0, 1]
        if self.mod == 2:  # 仅为了分类左右手想象
            if label == 2:
                return 1
            elif label == 3:
                return 0


def main(args):
    print("Generating training data.")
    data = generate_preprocess_dataset(path=args.path, output_dir=args.output_dir,
                                       class_mod=args.class_mod, samples=480, divide=args.divide)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="D:/data/physionet/MNE-eegbci-data/physiobank/database/eegmmidb/", help="data path"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data/generate/", help="path to save, None 不保存"
    )
    parser.add_argument(
        "--divide", type=list, nargs='+',default=[8, 1, 1], help="划分比例"
    )
    parser.add_argument(
        "--class_mod", type=int, default=2, help="分类模式 2 or 4"
    )
    parser.add_argument(
        "--lowpass", type=int,default=2, help=""
    )
    parser.add_argument(
        "--highpass", type=int,default=40, help=""
    )
    '''
    parser.add_argument(
        "--bandpass", type=list, nargs='+', default=[2,60], help="带通滤波频率"
    )
    '''
    args = parser.parse_args()
    main(args)
