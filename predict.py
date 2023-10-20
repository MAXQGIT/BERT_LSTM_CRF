# coding=utf-8

import os
import torch
from config import Config
from utils import transform_sentence, load_label_dict

def Checkpoint(path):
    read_path = os.path.join(path,'checkpoint')
    for i in open(read_path,'r',encoding='utf-8').readlines():
        if 'epoch' in i:
            model_path = os.path.join(path,i)
            return model_path

def  read_data():
    data_list = [line.strip('\n')[:125] for line in open('data/model_test.txt','r',encoding='utf-8').readlines()]
    return data_list


if __name__ == '__main__':
    config = Config()
    # print(config.device)
    data_list = read_data()
    # sentence = '第五艘西班牙海军F-100级护卫舰即将装备集成通信控制系统。该系统由葡萄牙EID公司生产。该系统已经用于巴西海军的圣保罗航母，荷兰海军的四艘荷兰级海上巡逻舰和四艘西班牙海军BAM近海巡逻舰。F-105护卫舰于2009年初铺设龙骨。该舰预计2010年建造完成，2012年夏交付。'
    sentence = '第五艘西班牙海军F-100级护卫舰即将装备集成通信控制系统。该系统由葡萄牙EID公司生产。'
    for i in data_list[:100]:
        in_str, inputs, masks = transform_sentence(sentence)
        # 加载模型
        model_path = Checkpoint('result')
        # print(model_path)
        model = torch.load(model_path)
        # print(model)
        # model.to(config.device)
        model = model.to(config.device)
        inputs = inputs.to(config.device)
        masks = masks.to(config.device)
        # print('$$$$$$$$$$$$$$$$$$$$$$$$')
        # print(config.device)

        feats = model(inputs, masks)
        # print(model)

        path_score, best_path = model.crf(feats, masks)

        best_path_list = best_path.tolist()
        print(best_path_list)
        predict_result = [best_path_list[0][i] for i in range(len(in_str))]
        print('origin sentence: {}'.format(in_str))
        
        reverse_vocab = load_label_dict(config.label_file)
        print(predict_result)
        # print(sentence)
       # print(reverse_vocab)
        predict_result = [reverse_vocab[i] for i in predict_result]
        print('predict result: {}'.format(predict_result))
    
        print('=='*100)

