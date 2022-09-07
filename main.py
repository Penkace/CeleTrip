import pandas as pd
import os
import numpy as np
import re
import time
import datetime
from torch.utils import data
import tqdm
import random
import dgl
import dgl.data
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
import dgl.nn.pytorch as dglnn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import math
from dgl.nn import GraphConv
import dgl.function as fn
from dgl import DGLGraph
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')
import scipy
from sklearn.model_selection import ParameterSampler

from models.celetrip import *
from dataset.dataset import *

from matplotlib import pyplot as plt
import os
os.environ['CUDA_LAUNCH_BLOCKING']='0'

import pickle
def save_pkl(path,obj):
    with open(path, 'wb') as f:
        pickle.dump(obj,f)
        
def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

class Config:
    device = 'cuda:0'
    word_embeddings_path = '/public/home/pengkai/Itinerary_Miner/Article_contain_person_name/News/all_trip_dataset/trip_w2v_model.model'
    
    #------------------------------------------------------------------------------------#
    # save_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/graph_data/path_with_event_entity/'
    
    # train_semantic_graph_path  = save_path + 'train_semantic_graph/'
    # test_semantic_graph_path = save_path + 'test_semantic_graph/'
    
    # train_semantic_properties = save_path + 'train_properties.pkl'
    # test_semantic_properties = save_path + 'test_properties.pkl'
    
    # train_sub_graph_save_path =save_path + 'train_sub_graph/'
    # test_sub_graph_save_path = save_path +'test_sub_graph/'
    
    # train_sub_graph_properties_path = save_path + 'train_sub_properties_path/'
    # test_sub_graph_properties_path = save_path + 'test_sub_properties_path/'
    
    # train_entity_save_path = save_path + 'train_entity_path/'
    # train_event_save_path = save_path + 'train_event_path/'
    # train_event_sent_save_path = save_path + 'train_event_sen_path/'
    
    # test_entity_save_path = save_path + 'test_entity_path/'
    # test_event_save_path = save_path + 'test_event_path/'
    # test_event_sent_save_path = save_path + 'test_event_sen_path/'
    
    # best_model_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/best_model/best_model_15_1.bin'
    # final_best_model_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/best_model/final_best_model_15_1.bin'
    # pic_save_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/pic/train_val_loss_15_1.png'
    # event_file = '/public/home/pengkai/Itinerary_Miner/CeleTrip/sentence_important/event_sentence_weight_15_1.pkl'
    # log_file = './log/log_15_1.txt'
    # best_predict_save_file = '/public/home/pengkai/Itinerary_Miner/CeleTrip/result/result_15_1.csv'

    # best_model_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/best_model/best_model_1_64.bin'
    # final_best_model_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/best_model/final_best_model_1_64.bin'
    # pic_save_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/pic/train_val_loss_1_64.png'
    # event_file = '/public/home/pengkai/Itinerary_Miner/CeleTrip/sentence_important/event_sentence_weight_1_64.pkl'
    # log_file = './log/log_1_64.txt'
    # best_predict_save_file = '/public/home/pengkai/Itinerary_Miner/CeleTrip/result/result_1_64.csv'

    # best_model_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/best_model/best_model_1_64_1.bin'
    # final_best_model_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/best_model/final_best_model_1_64_1.bin'
    # pic_save_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/pic/train_val_loss_1_64_1.png'
    # event_file = '/public/home/pengkai/Itinerary_Miner/CeleTrip/sentence_important/event_sentence_weight_1_64_1.pkl'
    # log_file = './log/log_1_64_1.txt'
    # best_predict_save_file = '/public/home/pengkai/Itinerary_Miner/CeleTrip/result/result_1_64_1.csv'

    # best_model_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/best_model/best_model_1_avgpooling.bin'
    # final_best_model_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/best_model/final_best_model_1_avgpooling.bin'
    # pic_save_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/pic/train_val_loss_1_avgpooling.png'
    # event_file = '/public/home/pengkai/Itinerary_Miner/CeleTrip/sentence_important/event_sentence_weight_1_avgpooling.pkl'
    # log_file = './log/log_1_avgpooling.txt'
    # best_predict_save_file = '/public/home/pengkai/Itinerary_Miner/CeleTrip/result/result_1_avgpooling.csv'
    #------------------------------------------------------------------------------------#

    #------------------------------------------------------------------------------------#
    # save_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/graph_data/path_with_text/'
    
    # train_semantic_graph_path  = save_path + 'train_semantic_graph/'
    # test_semantic_graph_path = save_path + 'test_semantic_graph/'
    
    # train_semantic_properties = save_path + 'train_properties.pkl'
    # test_semantic_properties = save_path + 'test_properties.pkl'
    
    # train_sub_graph_save_path =save_path + 'train_sub_graph/'
    # test_sub_graph_save_path = save_path +'test_sub_graph/'
    
    # train_sub_graph_properties_path = save_path + 'train_sub_properties_path/'
    # test_sub_graph_properties_path = save_path + 'test_sub_properties_path/'
    
    # train_entity_save_path = save_path + 'train_entity_path/'
    # train_event_save_path = save_path + 'train_event_path/'
    # train_event_sent_save_path = save_path + 'train_event_sen_path/'
    
    # test_entity_save_path = save_path + 'test_entity_path/'
    # test_event_save_path = save_path + 'test_event_path/'
    # test_event_sent_save_path = save_path + 'test_event_sen_path/'

    # best_model_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/best_model/best_model_with_text.bin'
    # final_best_model_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/best_model/final_best_model_with_text.bin'
    # pic_save_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/pic/train_val_loss_with_text.png'
    # event_file = '/public/home/pengkai/Itinerary_Miner/CeleTrip/sentence_important/event_sentence_weight_with_text.pkl'
    # log_file = './log/log_with_text.txt'
    # best_predict_save_file = '/public/home/pengkai/Itinerary_Miner/CeleTrip/result/result_with_text.csv'
    #------------------------------------------------------------------------------------#

    #------------------------------------------------------------------------------------#
    # save_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/graph_data/path_without_entity/'
    
    # train_semantic_graph_path  = save_path + 'train_semantic_graph/'
    # test_semantic_graph_path = save_path + 'test_semantic_graph/'
    
    # train_semantic_properties = save_path + 'train_properties.pkl'
    # test_semantic_properties = save_path + 'test_properties.pkl'
    
    # train_sub_graph_save_path =save_path + 'train_sub_graph/'
    # test_sub_graph_save_path = save_path +'test_sub_graph/'
    
    # train_sub_graph_properties_path = save_path + 'train_sub_properties_path/'
    # test_sub_graph_properties_path = save_path + 'test_sub_properties_path/'
    
    # train_entity_save_path = save_path + 'train_entity_path/'
    # train_event_save_path = save_path + 'train_event_path/'
    # train_event_sent_save_path = save_path + 'train_event_sen_path/'
    
    # test_entity_save_path = save_path + 'test_entity_path/'
    # test_event_save_path = save_path + 'test_event_path/'
    # test_event_sent_save_path = save_path + 'test_event_sen_path/'

    # best_model_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/best_model/best_model_without_entity.bin'
    # final_best_model_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/best_model/final_best_model_without_entity.bin'
    # pic_save_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/pic/train_val_loss_without_entity.png'
    # event_file = '/public/home/pengkai/Itinerary_Miner/CeleTrip/sentence_important/event_sentence_weight_without_entity.pkl'
    # log_file = './log/log_without_entity.txt'
    # best_predict_save_file = '/public/home/pengkai/Itinerary_Miner/CeleTrip/result/result_without_entity.csv'
    #------------------------------------------------------------------------------------#

    #------------------------------------------------------------------------------------#
    # save_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/graph_data/path_without_event/'
    
    # train_semantic_graph_path  = save_path + 'train_semantic_graph/'
    # test_semantic_graph_path = save_path + 'test_semantic_graph/'
    
    # train_semantic_properties = save_path + 'train_properties.pkl'
    # test_semantic_properties = save_path + 'test_properties.pkl'
    
    # train_sub_graph_save_path =save_path + 'train_sub_graph/'
    # test_sub_graph_save_path = save_path +'test_sub_graph/'
    
    # train_sub_graph_properties_path = save_path + 'train_sub_properties_path/'
    # test_sub_graph_properties_path = save_path + 'test_sub_properties_path/'
    
    # train_entity_save_path = save_path + 'train_entity_path/'
    # train_event_save_path = save_path + 'train_event_path/'
    # train_event_sent_save_path = save_path + 'train_event_sen_path/'
    
    # test_entity_save_path = save_path + 'test_entity_path/'
    # test_event_save_path = save_path + 'test_event_path/'
    # test_event_sent_save_path = save_path + 'test_event_sen_path/'

    # # best_model_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/best_model/best_model_without_event.bin'
    # # final_best_model_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/best_model/final_best_model_without_event.bin'
    # # pic_save_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/pic/train_val_loss_without_event.png'
    # # event_file = '/public/home/pengkai/Itinerary_Miner/CeleTrip/sentence_important/event_sentence_weight_without_event.pkl'
    # # log_file = './log/log_without_event.txt'
    # # best_predict_save_file = '/public/home/pengkai/Itinerary_Miner/CeleTrip/result/result_without_event.csv'

    # # best_model_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/best_model/best_model_without_event_128.bin'
    # # final_best_model_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/best_model/final_best_model_without_event_128.bin'
    # # pic_save_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/pic/train_val_loss_without_event_128.png'
    # # event_file = '/public/home/pengkai/Itinerary_Miner/CeleTrip/sentence_important/event_sentence_weight_without_event_128.pkl'
    # # log_file = './log/log_without_event_128.txt'
    # # best_predict_save_file = '/public/home/pengkai/Itinerary_Miner/CeleTrip/result/result_without_event_128.csv'


    # best_model_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/best_model/best_model_without_event_64.bin'
    # final_best_model_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/best_model/final_best_model_without_event_64.bin'
    # pic_save_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/pic/train_val_loss_without_event_64.png'
    # event_file = '/public/home/pengkai/Itinerary_Miner/CeleTrip/sentence_important/event_sentence_weight_without_event_64.pkl'
    # log_file = './log/log_without_event_64.txt'
    # best_predict_save_file = '/public/home/pengkai/Itinerary_Miner/CeleTrip/result/result_without_event_64.csv'
    #------------------------------------------------------------------------------------#


    # -----------------------------------------------------------------------------------#
    '''
    去掉事件和实体
    '''
    save_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/graph_data/text_event_entity/'
    
    train_semantic_graph_path  = save_path + 'train_semantic_graph/'
    test_semantic_graph_path = save_path + 'test_semantic_graph/'
    
    train_semantic_properties = save_path + 'train_properties.pkl'
    test_semantic_properties = save_path + 'test_properties.pkl'
    
    train_sub_graph_save_path =save_path + 'train_sub_graph/'
    test_sub_graph_save_path = save_path +'test_sub_graph/'
    
    train_sub_graph_properties_path = save_path + 'train_sub_properties_path/'
    test_sub_graph_properties_path = save_path + 'test_sub_properties_path/'
    
    train_entity_save_path = save_path + 'train_entity_path/'
    train_event_save_path = save_path + 'train_event_path/'
    train_event_sent_save_path = save_path + 'train_event_sen_path/'
    
    test_entity_save_path = save_path + 'test_entity_path/'
    test_event_save_path = save_path + 'test_event_path/'
    test_event_sent_save_path = save_path + 'test_event_sen_path/'

    best_model_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/best_model/best_model.bin'
    pic_save_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/pic/train_loss_fig_5_5_5_5.png'
    log_file = './log/log_test.txt'
    event_file = '/public/home/pengkai/Itinerary_Miner/CeleTrip/sentence_important/sentence_event_text_entity_weight.pkl'
    save_file_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/result2/pred_text_entity_event_.csv'
    save_result_file = '/public/home/pengkai/Itinerary_Miner/CeleTrip/result2/output_prob_text_event_entity.pkl'
    final_save_best_model = '/public/home/pengkai/Itinerary_Miner/CeleTrip/best_model/final_best_model.bin'
    #------------------------------------------------------------------------------------#
    # 定义模型的参数
    output_dim = 2
    '''
    主图的输入输出维度
    '''
    mg_in_dim = 64
    mg_out_dim=64
    '''
    地点图的输入输出维度
    '''
    sg_in_dim = 100
    sg_out_dim =64
    '''
    人名的输入输出维度
    '''
    na_in_dim = 100
    na_out_dim = 64
    '''
    事件的输入、隐藏、输出层
    '''
    ev_in_dim = 100
    ev_hid_dim = 64
    ev_out_dim = 64
    '''
    实体的输入、输出层
    '''
    en_in_dim = 100
    en_out_dim=64

    num_layers = 2
    num_heads = 12
    feat_drops = 0
    attn_drops = 0
    alphas = 0.2
    residuals = False
    agg_modes = 'flatten'
    epoch=1000
    learning_rate = 0.001
    step_size = 50
    lr_scheduler_gamma=0.5
    patience = 10
    weight_decay = 5e-4
    #------------------------------------------------------------------------------------#

config = Config()
device = config.device

print('The parameters are : ',config.mg_in_dim,
                            config.mg_out_dim,
                            config.sg_in_dim,
                            config.sg_out_dim,
                            config.na_in_dim,
                            config.na_out_dim,
                            config.ev_in_dim,
                            config.ev_hid_dim,
                            config.ev_out_dim,
                            config.en_in_dim,
                            config.en_out_dim,
                            config.num_layers,
                            config.num_heads)


'''
声明保存日志
'''
import logging
file_path = config.log_file
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())# 打印时间
fh = logging.FileHandler(file_path)# 创建日志文件
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

random_seed = 42
def seed_everything(seed=random_seed):
    '''
    固定随机种子
    :param random_seed: 随机种子数目
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(random_seed)



def train(model,data,train_sampler,optimizer,criterion,epoch):
    model.train()
    running_loss = 0
    total_iters = len(train_dataloader)
    for i,idx in enumerate(train_sampler):
        batch = data[idx]
        labels = batch[1].to(device)
        main_graph = batch[0].to(device)
        sub_graph_list = batch[2]
        name_feature = batch[3].to(device)
        event_feature=  batch[4]
        entity_feature = batch[5].to(device)
        event_freq_feature = batch[6]# 记录事件的前后频率
        entity_graph_list = batch[7]
        entity_graph_features_list = batch[8]
        location_embed_matrix = batch[9].to(device)
        outputs,_ = model(main_graph,sub_graph_list,name_feature,event_feature,entity_feature,event_freq_feature,entity_graph_list,entity_graph_features_list,location_embed_matrix,device)

        loss = criterion(outputs,labels)
        running_loss+=loss.item()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),norm_type=2,max_norm=20)
        optimizer.step()
    running_loss = running_loss / len(train_sampler)
    return running_loss


def eval_net(model,data,index_sampler,criterion,show=False):    
    model.eval()

    total = len(index_sampler)
    total_loss  = 0
    total_correct = 0
    total_predict =[]
    total_labels = []
    event_sentences_weight = []
    output_prob_list = []
    for i,idx in enumerate(index_sampler):

        batch = data[idx]
        labels = batch[1].to(device)
        main_graph = batch[0].to(device)
        sub_graph_list = batch[2]
        name_feature = batch[3].to(device)
        event_feature=  batch[4]
        entity_feature = batch[5].to(device)
        event_freq_feature = batch[6]
        entity_graph_list = batch[7]
        entity_graph_features_list= batch[8]
        location_embed_matrix = batch[9].to(device)
        outputs,eve_sen_weight = model(main_graph,sub_graph_list,name_feature,event_feature,entity_feature,event_freq_feature,entity_graph_list,entity_graph_features_list,location_embed_matrix,device)
        loss = criterion(outputs,labels)
        outputs_prob = outputs.clone().detach().cpu().numpy()
        output_prob_list+=list(outputs_prob)
        event_sentences_weight.append(eve_sen_weight)
        _,predicted = torch.max(outputs.data,1)
        predicted = list(predicted.cpu().numpy())
        total_predict+=predicted
        label_list = list(labels.cpu().numpy())
        total_labels+=label_list
        total_loss+=loss.item()
    model.train()
    avg_loss = total_loss / len(index_sampler)
    score = accuracy_score(total_labels, total_predict)# 计算验证集的准确率
    precision = precision_score(total_labels, total_predict,average='macro')
    recall = recall_score(total_labels, total_predict,average='macro')
    f1 = f1_score(total_labels, total_predict,average='macro')
    if show:
        print('The length of labels : ',len(total_labels))
        print('The length of prediction labels : ',len(total_predict))
        score = accuracy_score(total_labels, total_predict)# 计算验证集的准确率
        precision = precision_score(total_labels, total_predict,average='macro')
        recall = recall_score(total_labels, total_predict,average='macro')
        f1 = f1_score(total_labels, total_predict,average='macro')
        print('Final Accuracy : ',score)
        print('Final Precision : ',precision)
        print('Final Recall : ',recall)
        print('Final f1_score : ',f1)
        rep = classification_report(total_labels, total_predict,digits=6,target_names=['non trip location','trip location'])
        print('Report :\n',rep)
        logger.info('Classification Report : ')
        logger.info(rep)
    return avg_loss,total_predict,total_labels,event_sentences_weight,score,precision,recall,f1,output_prob_list

if __name__=='__main__':
    print('Start Train')
    train_dataloader = SyntheticDataset(config.train_semantic_properties,
                            config.train_semantic_graph_path,
                            config.train_sub_graph_properties_path,
                            config.train_sub_graph_save_path,
                            config.train_entity_save_path,
                            config.train_event_save_path,
                            config.word_embeddings_path,
                            )
    print('Start Test')
    test_dataloader = SyntheticDataset(config.test_semantic_properties,
                            config.test_semantic_graph_path,
                            config.test_sub_graph_properties_path,
                            config.test_sub_graph_save_path,
                            config.test_entity_save_path,
                            config.test_event_save_path,
                            config.word_embeddings_path,
                            )
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    best_predict = 0
    max_epoch = config.epoch
    cv_num = 20

    for cv in range(cv_num):
        CRLoss = nn.CrossEntropyLoss()
        model = CeleTrip(config.output_dim,
                            config.mg_in_dim,
                            config.mg_out_dim,
                            config.sg_in_dim,
                            config.sg_out_dim,
                            config.na_in_dim,
                            config.na_out_dim,
                            config.ev_in_dim,
                            config.ev_hid_dim,
                            config.ev_out_dim,
                            config.en_in_dim,
                            config.en_out_dim,
                            config.num_layers,
                            config.num_heads,
                            config.feat_drops,
                            config.attn_drops,
                            config.alphas,
                            config.residuals,
                            agg_modes=config.agg_modes)
        if torch.cuda.is_available()==True:
            model = model.to(config.device)
        optimizer = optim.Adam(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=config.step_size,gamma=config.lr_scheduler_gamma)
        
        ###############################################################
        '''
        CV
        '''
        num_samples = len(train_dataloader)
        num_train = int(num_samples*0.9)
        # 固定train, val设置随机的index
        train_sampler = SubsetRandomSampler(torch.arange(num_train))
        val_sampler = SubsetRandomSampler(torch.arange(num_train,num_samples))
        ###############################################################

        best_acc = 0.0
        early_stop = 0
        train_loss_list = []
        val_loss_list = []
        for ep in range(max_epoch):
            train_loss = train(model,train_dataloader,train_sampler,optimizer,CRLoss,ep)
        #     break
        # break
            scheduler.step()
            train_loss,train_predict,train_labels,train_event_sentences_weight,train_score,train_precision,train_recall,train_f1,_ = eval_net(model,train_dataloader,train_sampler,CRLoss)
            val_loss,val_predict,val_labels,val_event_sentences_weight,val_score,val_precision,val_recall,val_f1,_ = eval_net(model,train_dataloader,val_sampler,CRLoss)
            train_loss_list.append(train_score)
            val_loss_list.append(val_score)
            print('Train Loss : ',train_loss)
            print('Val Accuracy : ',val_score)
            if val_score>=best_acc:
                best_acc = val_score
                torch.save(model.state_dict(),config.best_model_path)
                print('###### Save the best model #######')
                early_stop = 0
            else:
                early_stop+=1
            torch.cuda.empty_cache()
            if early_stop>config.patience:
                break

        plt.figure(figsize=(10,6),dpi=300)
        plt.plot(train_loss_list,label='train')
        plt.plot(val_loss_list,label='val')
        plt.legend()
        plt.savefig(config.pic_save_path,bbox_inches='tight')


        #################################################################
        '''
        加载最佳的模型
        '''
        model.load_state_dict(torch.load(config.best_model_path))
        test_sampler = list(range(len(test_dataloader)))
        test_loss,test_predict,test_labels,test_event_sentences_weight,test_score,test_precision,test_recall,test_f1,outputs_prob = eval_net(model,test_dataloader,test_sampler,CRLoss,show=True)
        accuracy_list.append(test_score)
        precision_list.append(test_precision)
        recall_list.append(test_recall)
        f1_list.append(test_f1)
        #################################################################


        #################################################################
        '''
        保存整个流程中的最佳预测结果
        '''
        if test_score>best_predict:
            # ----------------------------保存最优的模型 ----------------------------#
            best_predict = test_score
            pred_result = pd.DataFrame(test_predict,columns=['pred'])
            pred_result['label'] = test_labels
            pred_result.to_csv(config.save_file_path,index=False,encoding='utf-8')
            
            # 事件的句子
            save_pkl(config.event_file,test_event_sentences_weight)

            outputs_prob = np.array(outputs_prob)
            save_pkl(config.save_result_file,outputs_prob)
            torch.save(model.state_dict(),config.final_save_best_model)
            # ----------------------------保存最优的模型 ----------------------------#
        

    
    accuracy_list = np.array(accuracy_list)
    precision_list = np.array(precision_list)
    recall_list = np.array(recall_list)
    f1_list = np.array(f1_list)
    print('The mean and std of accuracy : ',accuracy_list.mean(),accuracy_list.std())
    print('The mean and std of precision : ',precision_list.mean(),precision_list.std())
    print('The mean and std of recall : ',recall_list.mean(),recall_list.std())
    print('The mean and std of f1 : ',f1_list.mean(),f1_list.std())
    print('The parameters are : ',config.mg_in_dim,
                            config.mg_out_dim,
                            config.sg_in_dim,
                            config.sg_out_dim,
                            config.na_in_dim,
                            config.na_out_dim,
                            config.ev_in_dim,
                            config.ev_hid_dim,
                            config.ev_out_dim,
                            config.en_in_dim,
                            config.en_out_dim,
                            config.num_layers,
                            config.num_heads)
    logger.info('result: \n')
    logger.info('The averge accuracy {:.6f}'.format(accuracy_list.mean()))
    logger.info('The averge precision {:.6f}'.format(precision_list.mean()))
    logger.info('The averge recall {:.6f}'.format(recall_list.mean()))
    logger.info('The averge f1score {:.6f}'.format(f1_list.mean()))