# -*-coding:utf-8-*-
"""
训练WSS_CMER
"""

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np

from torch.nn import functional as F
from torch.backends import cudnn
from torch.autograd import Variable

from STSDataLoader_only_tumor import STSDataLoader as STSDataLoderTumor
from STSDataLoader_with_pixel_label import STSDataLoader
from unet_cam import UNetCAM
from transform import All
from CAM_util import max_norm
from Loss import CombineLoss
from utils import EarlyStopping

device_id = 1


transform = All().cuda(device=device_id)
combine_loss = CombineLoss().cuda(device=device_id)

back_ratio = 1

def ds_fusion(ct_predict, pet_predict):
    ct_predict_0 = 1 - ct_predict
    pet_predict_0 = 1 - pet_predict
    Q = 1 - (ct_predict_0 * pet_predict) - (ct_predict * pet_predict_0)
    fusion = (ct_predict * pet_predict) / Q
    fusion = np.nan_to_num(fusion, nan=0)
    return fusion

def train(data_loader, model_list, optimizer_list, num_models=2):
    """
    模型训练的具体方法
    :param data_loader:
    :param net:
    :param scheduler:
    :param early_stopping:
    :param criterion
    :return:
    """

    tbar = tqdm(data_loader, ascii=True, desc='train', dynamic_ncols=True)
    for batch_idx, (case_id, pet_data, ct_data, label, image_label) in enumerate(tbar):
        pet_data = pet_data.cuda(device=device_id)
        ct_data = ct_data.cuda(device=device_id)
        label = label.type(torch.LongTensor)
        label = label.cuda(device=device_id)
        image_label = image_label.cuda(device=device_id)
        y = image_label.squeeze(1)
        input = torch.cat((ct_data, pet_data), dim=1)

        preds1 = []
        cams1 = []

        for i in range(num_models):
            pred1, cam1 = model_list[i](input[:, i:i + 1, :, :])
            pred1 = torch.squeeze(pred1, -1)
            pred1 = torch.squeeze(pred1, -1)
            preds1.append(pred1)
            cams1.append(max_norm(cam1))


        preds1_back = []

        for i in range(num_models):
            predict = cams1[i][ :, 1, :, :].unsqueeze(1)
            data_back = input[:, i:i + 1, :, :] * (1 - predict)
            pred1_back, _ = model_list[i](data_back)
            pred1_back = torch.squeeze(pred1_back, -1)
            pred1_back = torch.squeeze(pred1_back, -1)
            preds1_back.append(pred1_back)



        x2, cams1 = transform(input, cams1)

        preds2 = []
        cams2 = []

        for i in range(num_models):
            pred2, cam2 = model_list[i](x2[:, i:i + 1, :, :])
            pred2 = torch.squeeze(pred2, -1)
            pred2 = torch.squeeze(pred2, -1)
            preds2.append(pred2)
            cams2.append(max_norm(cam2))



        #计算CT分支的Loss
        ct_loss = combine_loss(preds1, cams1, preds1_back, preds2, cams2, y, 0)
        optimizer_list[0].zero_grad()
        ct_loss.backward()
        optimizer_list[0].step()

        #计算PET分支的Loss
        pet_loss = combine_loss(preds1, cams1, preds1_back, preds2, cams2, y, 1)
        optimizer_list[1].zero_grad()
        pet_loss.backward()
        optimizer_list[1].step()



        tbar.set_postfix({"ct_loss":ct_loss.item(), "pet_loss": pet_loss.item()})
        tbar.update(1)


def evaluate(data_loader, model_list,  type,  num_models=2):
    """
    验证数据集的方法
    :param data_loader:
    :param net:
    :param criterion:
    :param type:
    :return:
    """
    tbar = tqdm(data_loader, ascii=True, desc="[EVAL]{}".format(type), dynamic_ncols=True)
    anchor_case_id = -1
    predicts = []
    ct_predicts = []
    pet_predicts = []
    labels = []
    ct_loss_list = []
    pet_loss_list = []
    dice_list = []
    ct_dice_list = []
    pet_dice_list = []
    recall_list = []
    precision_list = []
    case_list = []

    for batch_idx, (case_id, pet_data, ct_data, label, image_label) in enumerate(tbar):
        pet_data = pet_data.cuda(device=device_id)
        ct_data = ct_data.cuda(device=device_id)
        label = label.type(torch.LongTensor)
        label = label.cuda(device=device_id)
        image_label = image_label.cuda(device=device_id)
        y = image_label.squeeze(1)
        input = torch.cat((ct_data, pet_data), dim=1)

        preds1 = []
        cams1 = []

        for i in range(num_models):
            pred1, cam1 = model_list[i](input[:, i:i + 1, :, :])
            pred1 = torch.squeeze(pred1, -1)
            pred1 = torch.squeeze(pred1, -1)
            preds1.append(pred1)
            cams1.append(max_norm(cam1))

        ct_predict = cams1[0][:, 1, :, :]
        pet_predict = cams1[1][:, 1, :, :]
        ct_image_predict = torch.softmax(preds1[0], dim=1).unsqueeze(-1).unsqueeze(-1)
        pet_image_predict = torch.softmax(preds1[1], dim=1).unsqueeze(-1).unsqueeze(-1)
        ct_predict = ct_predict.cpu().detach().numpy()
        pet_predict = pet_predict.cpu().detach().numpy()
        ct_image_predict = ct_image_predict.cpu().detach().numpy()
        pet_image_predict = pet_image_predict.cpu().detach().numpy()

        preds1_back = []

        for i in range(num_models):
            predict = cams1[i][:, 1, :, :].unsqueeze(1)
            data_back = input[:, i:i + 1, :, :] * (1 - predict)
            pred1_back, _ = model_list[i](data_back)
            pred1_back = torch.squeeze(pred1_back, -1)
            pred1_back = torch.squeeze(pred1_back, -1)
            preds1_back.append(pred1_back)

        x2, cams1 = transform(input, cams1)

        preds2 = []
        cams2 = []

        for i in range(num_models):
            pred2, cam2 = model_list[i](x2[:, i:i + 1, :, :])
            pred2 = torch.squeeze(pred2, -1)
            pred2 = torch.squeeze(pred2, -1)
            preds2.append(pred2)
            cams2.append(max_norm(cam2))

        ct_loss = combine_loss(preds1, cams1, preds1_back, preds2, cams2, y, 0)
        pet_loss = combine_loss(preds1, cams1, preds1_back, preds2, cams2, y, 1)





        predict = ds_fusion(ct_predict, pet_predict)

        predict[predict >= 0.5] = 1
        predict[predict < 0.5] = 0

        ct_predict[ct_predict >= 0.5] = 1
        ct_predict[ct_predict < 0.5] = 0

        pet_predict[pet_predict >= 0.5] = 1
        pet_predict[pet_predict < 0.5] = 0


        ct_loss_list.append(ct_loss.item())
        pet_loss_list.append(pet_loss.item())
        label = label.cpu().detach().numpy()

        for i in range(len(case_id)):
            case_id_item = case_id[i]
            predict_item = predict[i]
            ct_predict_item = ct_predict[i]
            pet_predict_item = pet_predict[i]
            label_item = label[i]
            if anchor_case_id != -1 and anchor_case_id != case_id_item:
                predict_array = np.stack(predicts, axis=0)
                label_array = np.stack(labels, axis=0)
                ct_predicts_array = np.stack(ct_predicts, axis=0)
                pet_predicts_array = np.stack(pet_predicts, axis=0)
                dice = 2 * (predict_array * label_array).sum() / (predict_array.sum() + label_array.sum())
                ct_dice = 2 * (ct_predicts_array * label_array).sum() / (ct_predicts_array.sum() + label_array.sum())
                pet_dice = 2 * (pet_predicts_array * label_array).sum() / (pet_predicts_array.sum() + label_array.sum())
                recall = (predict_array[label_array == 1] == 1).sum() / (label_array == 1).sum()
                precision = (predict_array[label_array == 1] == 1).sum() / ((predict_array == 1).sum()+0.001)
                dice_list.append(dice)
                ct_dice_list.append(ct_dice)
                pet_dice_list.append(pet_dice)
                recall_list.append(recall)
                precision_list.append(precision)
                case_list.append(anchor_case_id)
                predicts.clear()
                ct_predicts.clear()
                pet_predicts.clear()
                labels.clear()
                # print(anchor_case_id, case_id_item, dice)
            anchor_case_id = case_id_item
            predicts.append(predict_item)
            labels.append(label_item)
            ct_predicts.append(ct_predict_item)
            pet_predicts.append(pet_predict_item)


        tbar.set_postfix({"pet_loss":pet_loss.item(), "ct_loss": ct_loss.item()})
        tbar.update(1)

    predict_array = np.stack(predicts, axis=0)
    label_array = np.stack(labels, axis=0)
    ct_predicts_array = np.stack(ct_predicts, axis=0)
    pet_predicts_array = np.stack(pet_predicts, axis=0)
    dice = 2 * (predict_array * label_array).sum() / (predict_array.sum() + label_array.sum())
    ct_dice = 2 * (ct_predicts_array * label_array).sum() / (ct_predicts_array.sum() + label_array.sum())
    pet_dice = 2 * (pet_predicts_array * label_array).sum() / (pet_predicts_array.sum() + label_array.sum())
    recall = (predict_array[label_array == 1] == 1).sum() / (label_array == 1).sum()
    precision = (predict_array[label_array == 1] == 1).sum() / ((predict_array == 1).sum() + 0.001)
    dice_list.append(dice)
    ct_dice_list.append(ct_dice)
    pet_dice_list.append(pet_dice)
    recall_list.append(recall)
    precision_list.append(precision)
    case_list.append(anchor_case_id)
    # print(anchor_case_id, case_id, dice)

    for index in range(len(case_list)):
        print("case_id:{}, dice:{}, pet dice: {}, ct dice: {}, recall:{}, precision:{}".format(
            case_list[index], round(dice_list[index],3), round(pet_dice_list[index], 3), round(ct_dice_list[index], 3),
            round(recall_list[index],3), round(precision_list[index], 3)))

    dice = np.mean(np.array(dice_list))
    recall = np.mean(np.array(recall_list))
    precision = np.mean(np.array(precision_list))
    ct_loss = np.mean(np.array(ct_loss_list))
    pet_loss = np.mean(np.array(pet_loss_list))
    ct_dice = np.mean(np.array(ct_dice_list))
    pet_dice = np.mean(np.array(pet_dice_list))

    return (ct_loss, pet_loss, dice, recall, precision, ct_dice, pet_dice)



def main_shell(batch_size=1, num_gpu=1, lr=0.001, max_epoll=100):
    gpu_ids = [i for i in range(num_gpu)]
    torch.cuda.empty_cache()

    pet_net = UNetCAM(n_channels=1, n_classes=2).cuda(device=device_id)
    pet_net = pet_net.cuda(device=device_id)
    ct_net = UNetCAM(n_channels=1, n_classes=2).cuda(device=device_id)
    ct_net = ct_net.cuda(device=device_id)

    ct_optimizer = torch.optim.Adam(ct_net.parameters(), lr=lr, weight_decay=0.0005)
    ct_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        ct_optimizer, mode='min', factor=0.1, patience=3, verbose=True,
        threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08
    )
    pet_optimizer = torch.optim.Adam(pet_net.parameters(), lr=lr, weight_decay=0.0005)
    pet_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        pet_optimizer, mode='min', factor=0.1, patience=3, verbose=True,
        threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08
    )


    model_list = [ct_net, pet_net]
    optimizer_list = [ct_optimizer, pet_optimizer]

    early_stopping = EarlyStopping(patience=30, verbose=True)


    train_data = STSDataLoader(type="train", transform=None)
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    valid_train_data = STSDataLoderTumor(type="train")
    valid_train_data_loader = DataLoader(valid_train_data, batch_size=batch_size, shuffle=False)
    valid_data = STSDataLoderTumor(type="valid")
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    best_dice = 0

    for epoll in range(max_epoll):
        epoch_str = f' Epoch {epoll + 1}/{max_epoll} '
        print(f'{epoch_str:-^40s}')
        print(f'CT Learning rate: {ct_optimizer.param_groups[0]["lr"]}')
        print(f'PET Learning rate: {pet_optimizer.param_groups[0]["lr"]}')

        ct_net.train()
        pet_net.train()
        torch.set_grad_enabled(True)
        train(data_loader=train_data_loader, model_list=model_list, optimizer_list=optimizer_list)

        ct_net.train()
        pet_net.train()
        torch.set_grad_enabled(False)
        (train_ct_loss, train_pet_loss, train_dice, train_recall, train_precision,
         train_ct_dice, train_pet_dice) = evaluate(data_loader=valid_train_data_loader, model_list=model_list, type="train")
        (valid_ct_loss, valid_pet_loss, valid_dice, valid_recall, valid_precision,
         valid_ct_dice, valid_pet_dice) = evaluate(data_loader=valid_data_loader, model_list=model_list, type="valid")
        ct_scheduler.step(train_ct_loss)
        pet_scheduler.step(train_pet_loss)
        early_stopping(valid_pet_loss+valid_ct_loss)

        print("Train loss : {}, Train Dice : {}".format(round(train_pet_loss+train_ct_loss, 6), round(train_dice, 3)))
        print("Valid loss : {}, Valid Dice : {}".format(round(valid_pet_loss+valid_ct_loss, 6), round(valid_dice, 3)))

        with open("result/log.log", "a+") as file:
          file.writelines("Epoll:{}, train loss: {}, train dice: {}, train pet dice: {}, train ct dice: {}, train recall: {}, train precision: {},"
                          "valid loss: {}, valid dice: {}, valid pet dice: {}, valid ct dice: {}, valid recall: {}, valid precision:{}\n".format(
              epoll+1, round(train_pet_loss+train_ct_loss, 6), round(train_dice, 3), round(train_pet_dice, 3), round(train_ct_dice, 3), round(train_recall, 3), round(train_precision, 3),
                       round(valid_pet_loss+valid_ct_loss, 6), round(valid_dice, 3), round(valid_pet_dice, 3), round(valid_ct_dice, 3), round(valid_recall, 3), round(valid_precision, 3)
          ))

        if best_dice < valid_dice:
            best_dice = valid_dice
            torch.save(model_list[0].state_dict(), "result/epoll_{}_ct.pkl".format(epoll+1))
            torch.save(model_list[1].state_dict(), "result/epoll_{}_pet.pkl".format(epoll + 1))

        if early_stopping.early_stop:
            print("Early stopping")
            break


if __name__ == '__main__':
    main_shell(batch_size=2, num_gpu=2, lr=0.0001)