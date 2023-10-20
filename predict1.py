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
    data_list = [line.strip('\n') for line in open('data/model_test.txt','r',encoding='utf-8').readlines()]
    return data_list


if __name__ == '__main__':
    config = Config()
    # print(config.device)
    # data_list = read_data()
    # 加载模型
    model_path = Checkpoint('result')
    print(model_path)
    # print(model_path)
    model = torch.load(model_path)  # ,map_location=torch.device('cpu')
    # print(model)
    # model.to(config.device)
    # model = model.to(config.device)

    # sentence = '澳大利亚授予萨伯技术公司一份合同，为澳大利亚皇家空军F/A-18大黄蜂飞机安装BOL干扰弹投放器并进行飞行试验。澳大利亚空军将是首家在F/A-18上验证BOL系统的客户，测试和试验合同为后继的生产合同开辟了道路。萨伯公司将为计划2004年第4季度开始的RAAF试飞提供支持，该试验是大黄蜂升级项目2.3阶段的组成部分，项目旨在提高该机的电子战自我保护能力。这一项目可能成为目前世界上最大的战斗机升级项目，试验成功后伴随而来的将会是为大量F/A-18装备BOL干扰弹投放器合同。F/A-18的配置是每架飞机带4套BOL，这将显著增强飞机抵御各种导弹威胁的能力，并提高飞机生存力。墨尔本的航空结构公司被选中进行武器挂架改装的研发工作。萨伯公司的BOL系统是一种能携带160个箔条/曳光弹包的先进干扰弹投放器，其携带能力是常规投放器的5倍多。该系统已装备在美国海军F-14熊猫、英国鹞GR7和狂风以及美国空军/空中国民警卫队的F-15鹰上。目前正在生产为美国空军/空中国民警卫队F-15鹰、EF-2000和瑞典JAS-39鹰狮装备的系统。'
    data_list =read_data()
    for sentence in data_list:
        sentence_list = sentence[:127].replace('。', '。\n').split('\n')
        for sentence in sentence_list:
            in_str, inputs, masks = transform_sentence(sentence)
            inputs = inputs.to(config.device)
            masks = masks.to(config.device)
            # print('$$$$$$$$$$$$$$$$$$$$$$$$')
            # print(config.device)

            feats = model(inputs, masks)
            # print(model)
            # feats = feats.cuda()
            path_score, best_path = model.crf(feats, masks)

            best_path_list = best_path.tolist()
            # print(best_path_list)
            predict_result = [best_path_list[0][i] for i in range(len(in_str))]
            print('origin sentence: {}'.format(in_str))
            print('================================')
            reverse_vocab = load_label_dict(config.label_file)
            # print(predict_result)
            # print(sentence)
            # print(reverse_vocab)
            predict_result = [reverse_vocab[i] for i in predict_result]
            print('predict result: {}'.format(predict_result))
            with open('result.txt','a',encoding='utf-8') as a:
                for i,j in zip(in_str,predict_result):
                    a.write('{} {}'.format(i,j))
                    a.write('\n')
                    print(i,j)



