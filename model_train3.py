import random

import pandas as pd
import numpy as np
import nibabel as nib  # 处理.nii类型图片
# import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm
from random import randint

from unet import UNet
from PIL.PngImagePlugin import PngImageFile
import warnings

warnings.filterwarnings("ignore")  # ignore warnings

random.seed(123)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用gpu加速


class DataProcessor:
    def __init__(self):
        # self.foreground = 255
        self.background = 0
        self.pixel = 512
        self.slice_resize = (self.pixel, self.pixel)
        # self.split_ratio = (0.7,0.1,0.2) # 训练集，验证集，测试集的比例

    def mask_one_hot(self,
                     img_arr):  # 将label（512，512）转化为标准的mask形式（512，512，class_num）,这里class_num设置为1，所以出来是（l.h.1）而不是（l,h,2）
        img_arr = np.expand_dims(img_arr, 2)  # (256,256)->(256,256,1), 注意这里传进来的不是img，而是label
        mask_shape = img_arr.shape
        mask1 = np.zeros(mask_shape)
        mask2 = np.zeros(mask_shape)
        mask1[img_arr > self.background] = 1  # foreground
        mask2[img_arr == self.background] = 1  # LV
        mask = np.concatenate([mask1, mask2], 2)  # (len,height,class_num = 4)
        # mask = mask1
        return mask

    def get_data(self):
        train, val, test = 16, 1, 2
        list_dir = os.listdir('mydata2/image')
        name_list = [item.split('_')[2] for item in list_dir]
        sort_names = sorted(list(set(name_list)))
        train_names, val_names, test_names = sort_names[0:train], sort_names[train:train + val], sort_names[
                                                                                                 train + val:train + val + test]
        trains = list()
        vals = list()
        tests = list()
        for dirname in list_dir:
            name = dirname.split('_')[2]
            if name in train_names:
                trains.append(dirname)
            elif name in val_names:
                vals.append(dirname)
            else:
                tests.append(dirname)

        train_list = list()
        for dirname in trains:
            img = sitk.ReadImage(f'mydata2/image/{dirname}')
            img = sitk.GetArrayFromImage(img)
            minimum = np.min(img)
            gap = np.max(img) - minimum
            img = (img - minimum) / gap * 255  # 0，1缩放
            label = sitk.ReadImage(f'mydata2/label/{dirname}')
            label = sitk.GetArrayFromImage(label)
            biz_type = dirname.split('_')[1]
            person_name = dirname.split('_')[2]
            MRI_type = dirname.split('_')[3].strip('.nii')
            for id in range(img.shape[0]):
                img1 = img[id, :, :]
                label1 = label[id, :, :]
                img1 = Image.fromarray(img1).convert('L')
                img_resize = img1.resize(self.slice_resize, 0)
                label1 = Image.fromarray(label1).convert('L')
                label_resize = label1.resize(self.slice_resize, 0)
                train_list.append([img_resize, label_resize, id, biz_type, person_name, MRI_type])

        val_list = list()
        for dirname in vals:
            img = sitk.ReadImage(f'mydata2/image/{dirname}')
            img = sitk.GetArrayFromImage(img)
            minimum = np.min(img)
            gap = np.max(img) - minimum
            img = (img - minimum) / gap * 255  # 0，1缩放
            label = sitk.ReadImage(f'mydata2/label/{dirname}')
            label = sitk.GetArrayFromImage(label)
            biz_type = dirname.split('_')[1]
            person_name = dirname.split('_')[2]
            MRI_type = dirname.split('_')[3].strip('.nii')
            for id in range(img.shape[0]):
                img1 = img[id, :, :]
                label1 = label[id, :, :]
                img1 = Image.fromarray(img1).convert('L')
                img_resize = img1.resize(self.slice_resize, 0)
                label1 = Image.fromarray(label1).convert('L')
                label_resize = label1.resize(self.slice_resize, 0)
                val_list.append([img_resize, label_resize, id, biz_type, person_name, MRI_type])

        test_list = list()
        for dirname in tests:
            img = sitk.ReadImage(f'mydata2/image/{dirname}')
            img = sitk.GetArrayFromImage(img)
            minimum = np.min(img)
            gap = np.max(img) - minimum
            img = (img - minimum) / gap * 255 # 0，1缩放
            label = sitk.ReadImage(f'mydata2/label/{dirname}')
            label = sitk.GetArrayFromImage(label)
            biz_type = dirname.split('_')[1]
            person_name = dirname.split('_')[2]
            MRI_type = dirname.split('_')[3].strip('.nii')
            for id in range(img.shape[0]):
                img1 = img[id, :, :]
                label1 = label[id, :, :]
                img1 = Image.fromarray(img1).convert('L')
                img_resize = img1.resize(self.slice_resize, 0)
                label1 = Image.fromarray(label1).convert('L')
                label_resize = label1.resize(self.slice_resize, 0)
                test_list.append([img_resize, label_resize, id, biz_type, person_name, MRI_type])
        return train_list, val_list, test_list

    def dice_score(self, fig1, fig2, class_value):
        '''
        计算某种特定像素级类别的DICE SCORE
        :param fig1:
        :param fig2:
        :param class_value:
        :return:
        '''
        fig1_class = fig1 == class_value
        fig2_class = fig2 == class_value
        A = np.sum(fig1_class)
        B = np.sum(fig2_class)
        AB = np.sum(fig1_class & fig2_class)
        if A + B == 0:
            return 1
        return 2 * AB / (A + B)


class MyDataset(Dataset):  #
    '''
    继承了torch.utils.data.Dataset,用于加载数据，后续载入神经网络中
    '''

    def __init__(self, data, TensorTransform):
        self.data = data
        self.TensorTransform = TensorTransform

    def __getitem__(self, item):  # 这个是Dataset类的关键函数，形成数据的最终形式，通过迭代的形式喂给后续的神经网络
        img, mask, id, biz_type, person_name, MRI_type = self.data[item]
        img_arr = np.asarray(img)
        img_arr = np.expand_dims(img_arr, 2)  # (512，512)->(512，512,1) # 实际图像矩阵
        mask = DataProcessor().mask_one_hot(np.asarray(mask))

        return self.TensorTransform(img_arr), self.TensorTransform(mask), torch.tensor(id)

    def __len__(self):
        return len(self.data)


class nn_processor:
    def __init__(self, train_loader, valid_loader=None):
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def train(self, net, lr=0.01, EPOCH=40, max_iter=500, save_iter=500, print_iter=100, first_iter=0,
              loss_func=nn.BCEWithLogitsLoss(), loss_func2=nn.MSELoss()):
        net = net.to(device)  # 加入gpu
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        i = 0
        # loss_train_list = list()
        # loss_valid_list = list()
        # iter_list = list()
        stop = False
        # create_folder('model_save')
        # create_folder('loss')
        for epoch in range(EPOCH):
            if stop == True:
                break
            for step, (x, y, y2) in enumerate(self.train_loader):
                # print(torch.mean(x))
                # print(torch.mean(y))
                # print(y2)

                x, y, y2 = x.to(device), y.to(device), y2.to(device)
                output1, output2 = net(x)
                # print(output.shape) # (batchsize,classnum,l,h)
                # print(y.shape)       # (batchsize,classnum,l,h)

                # print(type(output1),type(y),type(output2),type(y2))
                # print(loss_func(output1, y))
                # print(loss_func2(output2,y2))
                output1 = output1.to(torch.float)
                y = y.to(torch.float)
                output2 = output2.to(torch.float)
                y2 = y2.to(torch.float)
                loss = loss_func(output1, y) * 0.01 + loss_func2(output2, y2).to(torch.float) * 0.0001
                # loss = loss_func2(output2,y2).to(torch.float)
                # print(loss, type(loss))
                # print('\n')
                # loss = loss_func(output1, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                i += 1

                if i % print_iter == 0:
                    print(f'\n epoch:{epoch + 1}\niteration: {i + first_iter}')
                    if i == max_iter:  # 达到最大迭代，保存模型
                        stop = True
                        torch.save(net.state_dict(), f'model_save2/{i + first_iter}.pth')
                        print('\n model saved!')
                        break
                    if i % save_iter == 0:  # 临时保存
                        try:
                            os.remove(f'model_save2/{i + first_iter - save_iter}.pth')  # 日志回滚，只保留最新的模型
                        except:
                            pass
                        torch.save(net.state_dict(), f'model_save2/{i + first_iter}.pth')
                        print(f'\n model temp {i + first_iter} saved!')
                    for data in self.valid_loader:
                        x_valid, y_valid, slice_valid = data
                        x_valid, y_valid, slice_valid = x_valid.to(device), y_valid.to(device), slice_valid.to(device)
                        output1, output2 = net(x_valid)
                        valid_loss = loss_func(output1, y_valid)
                        # loss_train_list.append(float(loss))  # 每隔10个iter，记录一下当前train loss
                        # loss_valid_list.append(float(valid_loss))  # 每隔10个iter，记录一下当前valid loss
                        # iter_list.append(i + first_iter)  # 记录当前的迭代次数
                        print('\n train_loss:', float(loss))
                        print('\n -----valid_loss-----:', float(valid_loss))
                        break


if __name__ == '__main__':
    batch_size = 2  # 设置部分超参数
    class_num = 2

    dp = DataProcessor()
    train_list, valid_list, test_list = dp.get_data()  # 获取训练集，验证集，测试集上的数据（暂时以列表的形式）
    # def check(id):
    #     img, label = train_list[id][0], train_list[id][1]
    #     plt.imshow(Image.blend(img, label, 0.5))
    #     plt.show()
    #     plt.imshow(img)
    #     plt.show()
    #     print(train_list[id])
    # check(132)
    print(len(train_list), len(valid_list), len(test_list))
    TensorTransform = transforms.Compose([  # transform to figure, for further passing to nn
        transforms.ToTensor(),  # ToTensor会给灰度图像自动增添一个维度
    ])

    train_data = MyDataset(train_list, TensorTransform=TensorTransform)
    valid_data = MyDataset(valid_list, TensorTransform=TensorTransform)  # 从image2tentor
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                              num_workers=0)  # batch_size是从这里的DataLoader传递进去的
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True, num_workers=0)
    net = UNet(class_num)
    # net.load_state_dict(torch.load('model_save2/105000.pth'))

    unet_processor = nn_processor(train_loader, valid_loader)
    unet_processor.train(net, EPOCH=400, max_iter=200000, first_iter=0, lr=0.01)

    # def predict(net,target,slice_resize = (512,512)):
    #     '''
    #     给定模型和图片，以及网络预测所需要的resize，预测mask，返回mask矩阵
    #     :param net:
    #     :param target:
    #     :return:
    #     '''
    #     if type(target) == str:
    #         img_target = Image.open(target)
    #         origin_size = img_target.size
    #         img_arr = np.asarray(img_target.resize(slice_resize,0))
    #     elif type(target) == PngImageFile or type(target) ==Image.Image:
    #         origin_size = target.size
    #         img_arr = np.asarray(target.resize(slice_resize,0))
    #     elif type(target) == np.ndarray:
    #         origin_size = target.shape
    #         img_arr = np.asarray(Image.fromarray(target).resize(slice_resize,0))
    #     else:
    #         print('<target type error>')
    #         return False
    #     TensorTransform = transforms.Compose([  # transform to figure, for further passing to nn
    #         transforms.ToTensor(),  # ToTensor会给灰度图像自动增添一个维度
    #     ])
    #     img_tensor = TensorTransform(img_arr)
    #     img_tensor4d = img_tensor.unsqueeze(0)  # 只有把图像3维（1，256，256）扩展成4维（1，1，256，256）才能放进神经网络预测
    #     img_tensor4d = img_tensor4d.to(device)
    #
    #     # print(type(img_tensor4d), net(img_tensor4d))
    #     return img_tensor4d, net(img_tensor4d)
    #
    # for item in valid_list:
    #     img, label = item[0], item[1]
    #     img_tensor, pre = predict(net, img)
    #     y_predict_arr = pre[0].squeeze(0).squeeze(0).cpu().detach().numpy()
    #     y_true_arr = np.asarray(label)










