'''
@Penkace
2022/07/07
'''
import os
from random import shuffle
import pandas as pd
import numpy as np
import math

import torch.nn as nn
import torch
import torch.nn.functional as F

import dgl
from dgl.data import DGLDataset
import dgl.nn.pytorch as dglnn
from dgl.dataloading import GraphDataLoader
from dgl.nn import GraphConv
import dgl.function as fn
from dgl import DGLGraph
from torch.utils.data.sampler import SubsetRandomSampler

from gensim.models import Word2Vec, KeyedVectors
import gensim

import pickle
def save_pkl(path,obj):
    with open(path, 'wb') as f:
        pickle.dump(obj,f)
        
def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

class SyntheticDataset(DGLDataset):
    def __init__(self,semantic_properties_path,# 语义图的节点特征
                    semantic_graph_path,# 语义图的边
                    sub_graph_properties_path,# 地点子图的节点特征
                    sub_graph_save_path,# 地点子图的边
                    entity_save_path,# 实体
                    event_save_path,# 事件句子
                    sub_graph_oriented_path,
                    word_embeddings_path):# 词嵌入的保存路径
        self.semantic_properties_path = semantic_properties_path
        self.semantic_graph_path = semantic_graph_path
        
        self.sub_graph_properties_path = sub_graph_properties_path
        self.sub_graph_save_path = sub_graph_save_path
        
        self.entity_save_path = entity_save_path
        self.event_save_path = event_save_path
        
        self.word_embeddings_path = word_embeddings_path
        self.sub_graph_oriented_path = sub_graph_oriented_path
        self.event_freq_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/graph_data/properties/event_freq_features.pkl'
        self.ent_feature_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/data/ent_openke_feature.pkl'
        self.rel_feature_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/data/rel_openke_feature.pkl'
        self.ent_rel_path = '/public/home/pengkai/Itinerary_Miner/HeteroTrip/data/ent_rel_dict.pkl'
        self.location_embedding ='/public/home/pengkai/Itinerary_Miner/HeteroTrip/data/location_embedding_static.pkl'
        # self.history_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/data/person_history_all_date.pkl'
        super().__init__(name='synthetic')
    
    def process(self):
        word_embeddings_dim =100
        self.dim_feats = word_embeddings_dim
        self.gclasses = 2
        
        #--------------------------------------------------#
        '''
        读取词表
        '''
        model = gensim.models.Word2Vec.load(self.word_embeddings_path)
        word_vocab = model.wv.vocab
        #--------------------------------------------------#

        
        
        #--------------------------------------------------#
        '''
        加载实体和关系
        '''
        ent_rel_dict=  load_pkl(self.ent_rel_path)
        ent_features= load_pkl(self.ent_feature_path)
        rel_features = load_pkl(self.rel_feature_path)
        #--------------------------------------------------#

        #--------------------------------------------------#
        '''
        加载地点的静态表示
        '''
        location_embedding_static = load_pkl('/public/home/pengkai/Itinerary_Miner/HeteroTrip/data/location_embedding_static.pkl')
        #--------------------------------------------------#

        

        # ------------------------------------------------- #
        '''
        加载人名辞典
        '''
        name_index_dict = load_pkl('/public/home/pengkai/Itinerary_Miner/CeleTrip/data/name_index_dict.pkl')
        # ------------------------------------------------- #

        

        # ------------------------------------------------- #
        '''
        加载人名辞典
        '''
        # history_dict= load_pkl(self.history_path)
        # ------------------------------------------------- #

        '''
        先加载主图
        主图保存在
        '''
        self.graphs = []
        self.labels = []
        self.sub_graphs = []
        self.name_features = []
        self.entity_features = []
        self.event_features = []
        self.event_freq_features = []
        self.entity_graph_list =[]
        self.entity_graph_features = []
        self.location_embedding_static_list = []
        self.sub_graph_oriented_paths = []
        # self.history_list = []

        main_graph_properties = load_pkl(self.semantic_properties_path)
        main_cnt_list = main_graph_properties['graph_id'].tolist()
        main_graph_label_list = main_graph_properties['label'].tolist()
        main_graph_sub_graph_id= main_graph_properties['sub_graph_id'].tolist()
        main_graph_node_dict_list = main_graph_properties['node_dict'].tolist()
        main_graph_event_dict_list = main_graph_properties['event_dict'].tolist()
        main_graph_entity_dict_list = main_graph_properties['entity_dict'].tolist()
        main_graph_person_dict_list=  main_graph_properties['this_date_person_dict'].tolist()
        main_graph_date_list = main_graph_properties['date'].tolist()
        test_cnt = 0
        '''
        mcnt表示语义图的索引
        mlal表示语义图的标签
        msgl表示地点子图的索引
        mndl表示语义图的节点字典，序号——>地点/人/事件/实体
        medl表示语义图的事件字典，事件——>序号
        mendl表示语义图的实体字典，实体——>序号
        '''
        for i,(mcnt,mlal,msgl,mndl,medl,mendl,mgpdl,dal) in enumerate(zip(main_cnt_list, main_graph_label_list, main_graph_sub_graph_id, main_graph_node_dict_list, main_graph_event_dict_list, main_graph_entity_dict_list,main_graph_person_dict_list,main_graph_date_list)):
            
            '''
            每个行程语义图下的地点子图的信息读取
            包含graph_id, word_id, id_word, graph_node_num
            '''
            sub_graph_properties = load_pkl(self.sub_graph_properties_path+str(mcnt)+'.pkl')# 读取行程语义图的关系
            sub_graph_properties = sub_graph_properties.groupby(['graph_id'])# 按照语义图的索引来
            sub_graph_properties = dict(list(sub_graph_properties))
            
            
            sub_graph_list = []
            sub_graph_oriented_path_list = []
            '''
            一个行程语义图有多个地点子图，msgl保存的就是地点子图的索引
            '''
            for j in msgl:
                '''
                每个地点子图都有自己的索引，保存在msgl中
                '''
                sub_graph_edge = pd.read_csv(self.sub_graph_save_path+str(j)+'.csv')# 读取边的保存路径
                sub_graph_path = pd.read_csv(self.sub_graph_oriented_path+str(j)+'.csv')# 读取路径的保存路径，有name mid loc
                sub_graph_oriented_path = sub_graph_path.values
                sub_graph_oriented_path_list.append(sub_graph_oriented_path)

                src = sub_graph_edge['src'].to_numpy()
                dst = sub_graph_edge['dst'].to_numpy()
                sub_num_nodes = sub_graph_properties[j]['graph_node_num'].tolist()[0]
                id_word_map = sub_graph_properties[j]['id_word'].tolist()[0]
                sub_graph_features = []
                for kid in id_word_map:# 保存行程三元组图的节点特征
                    sub_graph_features.append(model.wv[id_word_map[kid]])
                sub_graph_features = np.array(sub_graph_features)
                try:
                    sub_graph = dgl.graph((src,dst),num_nodes=sub_num_nodes)
                except:
                    print(src)
                sub_graph = dgl.remove_self_loop(sub_graph)
                sub_graph = dgl.add_self_loop(sub_graph)
                sub_graph.ndata['attr'] = torch.Tensor(sub_graph_features)
                sub_graph_list.append(sub_graph)
            
                
            '''
            读取这一天下的实体和事件特征
            entity_features的维度是 entity_num x entity_dim
            event_features是dict，其中每个键是事件的index，值为 sen_num x sen_dim
            '''
            date_graph_entity_features = load_pkl(self.entity_save_path+str(mcnt)+'.pkl')
            sub_entity_graph_list = []
            sub_entity_feature_list = []
            '''
            读取知识子图数据
            '''
            for kv in mendl:

                rel_dict = ent_rel_dict[kv]
                src_list = rel_dict['src']
                dst_list = rel_dict['dst']
                rel_list = rel_dict['rel']
                rel_num = len(rel_list)
                rel_idx = 0
                node_idx = 1
                idx_dict= {}
                one_src_list = []
                one_dst_list = []
                rel_src_list = []
                rel_dst_list = []
                for i,(sidx,cidx) in enumerate(zip(src_list,dst_list)):
                    if cidx not in idx_dict:# 给每个cidx建立一个索引
                        idx_dict[cidx] = node_idx
                        node_idx+=1
                    one_src_list.append(0)
                    one_dst_list.append(idx_dict[cidx])
                    rel_src_list.append(rel_idx)
                    rel_dst_list.append(rel_idx+rel_num)
                    rel_idx+=1
#                 print('SRC List : ',one_src_list)
#                 print('DST List : ',one_dst_list)
#                 print('REL list : ',rel_src_list)
#                 print('REL list2 : ',rel_dst_list)
#                 print('The number of REL',rel_num)
                egraph = dgl.DGLGraph()
                egraph.add_nodes(node_idx)# 添加节点以及对应的边
                egraph.add_edges(one_src_list,one_dst_list)
                egraph.add_edges(one_src_list,one_dst_list)
                edge_type = torch.tensor(rel_src_list+rel_dst_list)
                in_deg = egraph.in_degrees(range(egraph.number_of_nodes())).float().numpy()
#                 print('In degree : ',in_deg)
                norm = in_deg**-0.5
                norm[np.isinf(norm)] = 0
                norm = torch.from_numpy(norm)
                egraph.ndata['xxx'] = norm
                egraph.apply_edges(lambda edges: {'xxx': edges.dst['xxx'] * edges.src['xxx']})
                edge_norm = egraph.edata.pop('xxx').squeeze()
                x_feat = []
                x_feat.append(ent_features[sidx])
                for cidx in idx_dict:
                    x_feat.append(ent_features[cidx])
                x_feat = np.array(x_feat)
                one_rel_feat = []
                for ridx in rel_list:
                    one_rel_feat.append(rel_features[ridx])
                one_rel_feat+=one_rel_feat
                one_rel_feat  = np.array(one_rel_feat)
                
                sub_entity_graph_list.append(egraph)
                sub_entity_feature_list.append([x_feat,one_rel_feat,edge_type,edge_norm])
                
            
            self.entity_graph_list.append(sub_entity_graph_list)
            self.entity_graph_features.append(sub_entity_feature_list)
            
            date_graph_event_features = load_pkl(self.event_save_path+str(mcnt)+'.pkl')
            event_features = []
            for keve in date_graph_event_features:
                event_features.append(torch.from_numpy(date_graph_event_features[keve]))
            
            event_freq_features = []
            date_graph_event_freq_features = load_pkl(self.event_freq_path)
            for keve in date_graph_event_freq_features:
                event_freq_features.append(torch.from_numpy(date_graph_event_freq_features[keve]))


            # --------------------------------------------------------------- # 
            name_features_list = []
            # name_features_history_list = []
            for nal in mgpdl:
                name_features_list.append(name_index_dict[nal])
                # one_person_history_list = history_dict[nal][dal]
                # name_features_history_list.append(one_person_history_list)            
            # --------------------------------------------------------------- # 
    
            ##########################################################
            '''
            将location的静态Embedding加入
            '''
            this_row_location_number = len(msgl)
            this_row_node_dict = list(mndl.keys())
            
            location_embedding = []
            for kloc in range(this_row_location_number):
                loc_key = mndl[kloc]
                location_embedding.append(location_embedding_static[loc_key])
            self.location_embedding_static_list.append(np.stack(location_embedding))
            ##########################################################


            #########################################################################################
            '''
            主图以同构图的方式连接
            msgl,mndl,medl,mendl  主图的子图列表，节点字典，事件节点，实体节点
            '''
            main_graph_edges = pd.read_csv(self.semantic_graph_path+str(mcnt)+'.csv')
            main_src = main_graph_edges['src'].to_numpy()
            main_dst = main_graph_edges['dst'].to_numpy()
            # main_graph_node_number = len(main_graph_edges['src'].unique())
            main_graph_node_number = len(mndl)
            # print('The max ',main_graph_edges['src'].max())
            # print('The main graph ',len(main_graph_edges['src'].unique()))
            # ------------------------------------------------------------------------------- #
            main_date_graph = dgl.graph((main_src,main_dst),num_nodes=main_graph_node_number)
            main_date_graph = dgl.remove_self_loop(main_date_graph)
            main_date_graph = dgl.add_self_loop(main_date_graph)
            # ------------------------------------------------------------------------------- #
            
            self.graphs.append(main_date_graph)
            self.labels.append(torch.LongTensor(mlal))
            self.sub_graphs.append(sub_graph_list)
            self.name_features.append(name_features_list)
            self.entity_features.append(torch.from_numpy(date_graph_entity_features))
            self.event_features.append(event_features)
            self.event_freq_features.append(event_freq_features)
            self.sub_graph_oriented_paths.append(sub_graph_oriented_path_list)
            # self.history_list.append(name_features_history_list)
            test_cnt+=1
            # if test_cnt>50:
            #     break
#         return self.graphs,self.labels,self.event_features,self.entity_features,self.name_features
    
    def __getitem__(self,i):
        return self.graphs[i],self.labels[i],np.array(self.sub_graphs[i]),self.name_features[i],self.event_features[i],self.entity_features[i],self.event_freq_features[i],self.entity_graph_list[i],np.array(self.entity_graph_features[i]),torch.from_numpy(np.array(self.location_embedding_static_list[i])),self.sub_graph_oriented_paths[i]
    
    def __len__(self):
        return len(self.graphs)

class Config:
    save_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/graph_data2/path_oriented_w10_text_entity_event_history/'
    
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
    
    train_sub_graph_oriented_path = save_path + 'train_sub_graph_oriented_path/'
    test_sub_graph_oriented_path = save_path + 'test_sub_graph_oriented_path/'

    word_embeddings_path = '/public/home/pengkai/Itinerary_Miner/Article_contain_person_name/News/all_trip_dataset/trip_w2v_model.model'

config = Config()
model = gensim.models.Word2Vec.load(config.word_embeddings_path)
word_vocab = model.wv.vocab




if __name__ =='__main__':
    dataloader = SyntheticDataset(config.train_semantic_properties,
                                    config.train_semantic_graph_path,
                                    config.train_sub_graph_properties_path,
                                    config.train_sub_graph_save_path,
                                    config.train_entity_save_path,
                                    config.train_event_save_path,
                                    config.train_sub_graph_oriented_path,
                                    config.word_embeddings_path,
                                    )
    
