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
        mask2[img_arr == self.background] = 0  # LV
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
        # df = pd.read_csv('check_match.csv')
        # data_list = list()
        # for i in tqdm(range(len(df))): # 遍历df每一行，添加进度条
        #     biz_type, person_name, MRI_type,abnormal_type,match_relation = df.loc[i, 'biz_type'],df.loc[i, 'person_name'],df.loc[i, 'MRI_type'],df.loc[i, 'abnormal_type'],df.loc[i, 'match_relation']
        #     if abnormal_type != 'normal' or match_relation == 'unknown' or MRI_type!= 'T1':
        #         continue # 如果是异常类型，则先不做处理,只对unknown进行处理
        #     img_path = 'liver_segmentation/image/{0}/{1}/{2}'.format(biz_type, person_name, MRI_type)
        #     label_path = 'liver_segmentation/label/{0}/{1}/{2}'.format(biz_type, person_name, MRI_type)
        #     img_list_dir = os.listdir(img_path)
        #     label_list_dir = os.listdir(label_path)
        #     dcm_file = sorted([item for item in img_list_dir if '.dcm' in item])  # 获取dcm后缀文件并进行排序
        #     dcm_num = len(dcm_file)  # 统计dcm文件的数量
        #     for item in label_list_dir:  # 获取label nii图像矩阵
        #         if '.nii' in item:
        #             label_nii = item  # 因此已经去除了异常点，所以一定能找到一个nii文件
        #             break
        #     label_array = nib.load(label_path + '/' + label_nii).get_fdata() * 255  # 得到label的三维数据矩阵,这个地方要放大一下，从nii过来的时候量纲比较小
        #     for id in range(dcm_num):
        #         # print('processing {0}_{1}_{2}_{3}'.format(person_name, MRI_type, id,match_relation))
        #         img = sitk.ReadImage(img_path + '/' + dcm_file[id])
        #         img = sitk.GetArrayFromImage(img)
        #         img = np.squeeze(img)
        #         img = np.rot90(img, -1)  # np矩阵顺时针旋转90°,得到目标图像矩阵
        #         img = Image.fromarray(img).convert('L')
        #         img_resize = img.resize(self.slice_resize, 0)
        #         transform1 = transforms.CenterCrop(self.slice_resize)
        #         random_rotate = randint(1, 360)  # 随机旋转角度
        #         if match_relation == 'ASC': # 升序
        #             label = label_array[:, :, id]
        #         else: # 降序
        #             label = label_array[:, :, dcm_num - id - 1]
        #         # if np.sum(label) < 10 * 255: # 为了防止梯度爆炸，把小于10个前景像素点的slice直接舍弃
        #         #     continue
        #         label = Image.fromarray(label).convert('L')
        #         label_resize = label.resize(self.slice_resize, 0)
        #
        #         data_list.append([img_resize,label_resize, id, biz_type,person_name, MRI_type])  # 样本一、resize
        #         # data_list.append([transform1(img), transform1(label), id, biz_type,person_name, MRI_type])  # 样本二、中心裁剪
        #         # data_list.append([img_resize.rotate(random_rotate), label_resize.rotate(random_rotate), id, biz_type,person_name, MRI_type])  # 样本三，随机旋转
        #         # data_list.append([img_resize.transpose(Image.FLIP_LEFT_RIGHT), label_resize.transpose(Image.FLIP_LEFT_RIGHT), id, biz_type,person_name, MRI_type]) # 样本四，左右旋转
        #         # data_list.append([img_resize.transpose(Image.FLIP_TOP_BOTTOM), label_resize.transpose(Image.FLIP_TOP_BOTTOM), id, biz_type,person_name, MRI_type])  # 样本五，上下旋转
        # # random.shuffle(data_list) # 随机打乱 # 这里暂时把随机性去掉了
        # train_split, valid_split, test_split = self.split_ratio
        # b1, b2 = int(len(data_list) * train_split), int(len(data_list) * (train_split + valid_split))
        # train_list, valid_list, test_list = data_list[0:b1], data_list[b1:b2],data_list[b2:] # 按照指定的比例划分表训练集，验证集，测试集
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
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
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
    model_list_dir = os.listdir('model_save2')
    max_number = max([int(item.strip('.pth')) for item in model_list_dir])
    net.load_state_dict(torch.load(f'model_save2/{max_number}.pth'))
    net = net.to(device)  # 加入gpu


    # # 训练
    # unet_processor = nn_processor(train_loader, valid_loader)
    # unet_processor.train(net, EPOCH=400, max_iter=200000, first_iter=0, lr=0.01)

    def predict(net,target,slice_resize = (512,512)):
        '''
        给定模型和图片，以及网络预测所需要的resize，预测mask，返回mask矩阵
        :param net:
        :param target:
        :return:
        '''
        if type(target) == str:
            img_target = Image.open(target)
            origin_size = img_target.size
            img_arr = np.asarray(img_target.resize(slice_resize,0))
        elif type(target) == PngImageFile or type(target) ==Image.Image:
            origin_size = target.size
            img_arr = np.asarray(target.resize(slice_resize,0))
        elif type(target) == np.ndarray:
            origin_size = target.shape
            img_arr = np.asarray(Image.fromarray(target).resize(slice_resize,0))
        else:
            print('<target type error>')
            return False
        TensorTransform = transforms.Compose([  # transform to figure, for further passing to nn
            transforms.ToTensor(),  # ToTensor会给灰度图像自动增添一个维度
        ])
        img_tensor = TensorTransform(img_arr)
        img_tensor4d = img_tensor.unsqueeze(0)  # 只有把图像3维（1，256，256）扩展成4维（1，1，256，256）才能放进神经网络预测
        img_tensor4d = img_tensor4d.to(device)

        # print(type(img_tensor4d), net(img_tensor4d))
        return img_tensor4d, net(img_tensor4d)

    y_pre_list = list()
    y_true_list = list()
    # 开始合并三维
    glist = list()
    bt = 'ximenzi'
    pt = 'zhangxiaomin'
    mt = 'DWI'
    lblb = sitk.ReadImage(f'mydata2/label/51_{bt}_{pt}_{mt}.nii')
    lblb = sitk.GetArrayFromImage(lblb)
    resize_shape = (lblb.shape[2], lblb.shape[1])

    for item in test_list:
        img, label = item[0], item[1]
        biz_type, person_name, MRI_type = item[3], item[4],item[5]
        img_tensor, pre = predict(net, img)
        y_predict_arr = pre[0].squeeze(0).squeeze(0).cpu().detach().numpy()
        y_true_arr = np.asarray(label)
        y_pre_list.append(y_predict_arr)
        y_true_list.append(y_true_arr)
        if biz_type == bt and person_name == pt and MRI_type == mt:
            img1 = y_predict_arr[1, :, :] < y_predict_arr[0, :, :]
            img1 = Image.fromarray(img1).convert('L')
            img_resize = img1.resize(resize_shape, 0)
            img_resize = np.asarray(img_resize)
            # img_resize = np.flipud(img_resize)
            # img_resize = np.fliplr(img_resize)
            img_resize = np.expand_dims(img_resize, 0)
            glist.append(img_resize/255)

    tmp = np.concatenate(glist, 0)
    tmp_simg = sitk.GetImageFromArray(tmp)
    sitk.WriteImage(tmp_simg, f'mask_practice.nii')
    print(sitk.GetArrayFromImage(tmp_simg).shape)
    print(DataProcessor().dice_score(lblb,sitk.GetArrayFromImage(tmp_simg),1))
    # id = 1
    # plt.imshow(y_pre_list[id][1, :, :] < y_pre_list[id][0, :, :])
    # ktmp = sitk.GetArrayFromImage(tmp_simg)[10, :, :]
    # plt.imshow(ktmp)










