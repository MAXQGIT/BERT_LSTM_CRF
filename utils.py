# coding=utf-8
import torch
import os
import datetime
import unicodedata
from config import Config


class InputFeatures(object):
    def __init__(self, input_id, label_id, input_mask):
        self.input_id = input_id
        self.label_id = label_id
        self.input_mask = input_mask


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = {}
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def read_data(path):
    data_0 = []
    with open(path, 'r', encoding='utf-8') as f:
        text, label = [], []
        for line in f.readlines():
            if len(line) > 1:
                text.append(line.strip('\n').split(' ')[0])
                label.append(line.strip('\n').split(' ')[1])
            else:
                data_0.append((text, label))
                text, label = [], []
    data = [i for i in data_0 if len(i[0]) > 1]
    return data


def read_corpus(path, max_length, label_dic, vocab):
    """
    :param path:数据文件路径
    :param max_length: 最大长度
    :param label_dic: 标签字典
    :return:
    """
    data = read_data(path)
    result = []
    for line in data:

        tokens = line[0]
        label = line[1]
        if len(tokens) > max_length - 2:
            tokens = tokens[0:(max_length - 2)]
            label = label[0:(max_length - 2)]
        tokens_f = ['[CLS]'] + tokens + ['[SEP]']
        label_f = ["<start>"] + label + ['<eos>']
        input_ids = [int(vocab[i]) if i in vocab else int(vocab['[UNK]']) for i in tokens_f]
        label_ids = [label_dic[i] for i in label_f]
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_length:
            input_ids.append(0)
            input_mask.append(0)
            label_ids.append(label_dic['<pad>'])

        assert len(input_ids) == max_length
        assert len(input_mask) == max_length
        assert len(label_ids) == max_length
        feature = InputFeatures(input_id=input_ids, input_mask=input_mask, label_id=label_ids)
        result.append(feature)
    return result


def save_model(model, epoch, path='result', **kwargs):
    """
    默认保留所有模型
    :param model: 模型
    :param path: 保存路径
    :param loss: 校验损失
    :param last_loss: 最佳epoch损失
    :param kwargs: every_epoch or best_epoch
    :return:
    """
    if not os.path.exists(path):
        os.mkdir(path)
    if kwargs.get('name', None) is None:
        # cur_time = datetime.datetime.now().strftime('%Y-%m-%d#%H:%M:%S')
        name = '--epoch_{}'.format(epoch)
        full_name = os.path.join(path, name)
        torch.save(model, full_name)
        print('Saved model at epoch {} successfully'.format(epoch))
        with open('{}/checkpoint'.format(path), 'w') as file:
            file.write(name)
            print('Write to checkpoint')


def load_model(model, path='result', **kwargs):
    if kwargs.get('name', None) is None:
        with open('{}/checkpoint'.format(path)) as file:
            content = file.read().strip()
            name = os.path.join(path, content)
    else:
        name = kwargs['name']
        name = os.path.join(path, name)
    model.load_state_dict(torch.load(name, map_location=lambda storage, loc: storage))
    print('load model {} successfully'.format(name))
    return model


def transform_sentence(sentence):
    """
    用于输入文本的转换 虽然加了列表其实是为了和模型匹配ヾ(≧▽≦*)
    :param sentence:自然输入的句子 单句 比如: 李白是我国著名的刺客
    :return in_ids:对照词汇表编码后的文本id列表
    :return in_masks:文本位置计数表
    """
    config = Config()
    vocab = load_vocab(config.vocab)
    max_length = config.max_length
    sentence_list = sentence.replace('。', '。\n').split('\n')
    # for sentence in sentence_list:
    in_str = []
    # 构造一个单字符列表
    for i in sentence:
        in_str.append(i)
    # 格式对齐
    in_str = ['[CLS]'] + in_str + ['[SEP]']
    in_ids = [int(vocab[i]) if i in vocab else int(vocab['[UNK]']) for i in in_str]
    in_masks = [1] * len(in_ids)
    while len(in_ids) < max_length:
        in_ids.append(0)
        in_masks.append(0)
    assert len(in_ids) == max_length
    assert len(in_masks) == max_length
    # 当前是两个list 需要转成tensor
    in_ids = torch.tensor([in_ids])
    in_masks = torch.tensor([in_masks])
    return in_str, in_ids, in_masks


def save_label_dict(data_path, save_path):
    label_list = []
    data = read_data(data_path)
    for line in data:
        label_list += line[1]
    label_list = ["<start>", "<pad>"] + list(set(label_list)) + ['<eos>']
    if os.path.exists(save_path):
        os.remove(save_path)
    with open(save_path, 'a', encoding='utf-8') as w:
        for word in label_list:
            w.write(word)
            w.write('\n')
    return label_list


def load_label_dict(path):
    label_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        index = 0
        for word in f.readlines():
            label_dict[index] = word.strip('\n')
            index += 1
    return label_dict


if __name__ == '__main__':
    data_path = 'data/train.txt'
    save_path = 'data/tag.txt'
    label_list = save_label_dict(data_path, save_path)
    vocab_file = 'data/tag.txt'
    vocab = load_vocab(vocab_file)
    print(vocab)
