import os
from dgl.nn.pytorch.glob import MaxPooling, AvgPooling, SumPooling
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

import networkx as nx
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from dgl.nn.pytorch import GATConv
'''
传入主图mg_dim
子图列表sg_dim
人名特征na_dim
事件特征ev_dim
实体特征en_dim
'''
class GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, feat_drop, attn_drop,
                 alpha=0.2, residual=True, agg_modes='flatten'):
        super(GATLayer, self).__init__()
        # 
        self.gat_conv = GATConv(in_feats=in_feats, out_feats=out_feats, num_heads=num_heads,
                                feat_drop=feat_drop, attn_drop=attn_drop,
                                negative_slope=alpha, residual=residual)
        assert agg_modes in ['flatten', 'mean']
        self.agg_modes = agg_modes

    def forward(self, bg, feats):
        feats = self.gat_conv(bg, feats)
        if self.agg_modes == 'flatten':
            feats = feats.flatten(1)
        else:
            feats = feats.mean(1)
        return feats


class GCNLayer(nn.Module):
    def __init__(self,in_feats,out_feats):
        super(GCNLayer,self).__init__()

        self.gcn_conv = GraphConv(in_feats,out_feats)

    def forward(self,bg,feats):
        feats = self.gcn_conv(bg,feats)
        return feats


class SemanticAttention(nn.Module):
    def __init__(self,in_size,hidden_size=128):
        super(SemanticAttention,self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size,hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size,1)
        )
    def forward(self,z):    
        w = self.project(z)            
        each_layer_weight = w.detach().cpu().numpy()

        weights = torch.softmax(w, dim=0)
        feats = torch.sum(weights*z,dim=0) 
        return feats,each_layer_weight

    
    

'''
outputs是输出的维度
mg_dim是行程语义图的特征维度
mg_out是行程语义图的隐藏层维度
'''
class CeleTrip(nn.Module):
    def __init__(self,outputs,
                 mg_dim,mg_out,
                 sg_dim,sg_hid,sg_out,
                 na_dim,na_out,
                 ev_dim,ev_hid,ev_out,
                 en_dim,en_out,
                 num_layers,num_heads,feat_drops,attn_drops,alphas,residuals,agg_modes,activations=None):
        super(CeleTrip,self).__init__()
        
        '''
        定义子图的学习
        '''
        self.sub_graph_oriented_aggregator = SemanticAttention(sg_dim,sg_hid)
        self.sub_graph_oriented_linear_layer = nn.Linear(sg_dim,sg_out)
        self.subg_list = nn.ModuleList()
        for i in range(num_layers):
            self.subg_list.append(GCNLayer(sg_dim,sg_out))
            sg_dim = sg_out        
        # self.sg_pooling = AvgPooling()
        self.sg_pooling = MaxPooling()
        self.sg_word_feats_linear_layer = nn.Linear(2*sg_out,sg_out)
        self.sg_out_dim = sg_out
        self.sg_linear = nn.Linear(sg_out,na_out)
        self.sg_linear2 = nn.Linear(na_out+50,na_out)# 聚集地点的特征
        
        '''
        定义事件的layer
        '''
        self.ea_layer = SemanticAttention(ev_dim,ev_hid)
        self.ea_linear_layer= nn.Linear(ev_dim,ev_out)
        self.ev_freq_layer = nn.Linear(15,ev_out)
        self.ev_freq_con = nn.Linear(2*ev_out,ev_out)


        '''
        定义实体的学习层
        '''
        self.en_layer = nn.Linear(en_dim,en_out)
        self.en_compgcn = CompGCNCov(in_channels=50,out_channels=en_out)
        self.en_dim = en_out

        '''
        人名的线性变换层
        '''
        self.na_layer = nn.Linear(na_dim,na_out)
        self.na_emb = nn.Embedding(50,100)
        
        '''
        行程语义图的学习层
        '''
        self.mg_list = nn.ModuleList()
        for i in range(num_layers):
            self.mg_list.append(GATLayer(mg_dim,mg_out,num_heads,feat_drops,attn_drops,alphas,residuals,agg_modes='flatten'))
            if agg_modes=='flatten':
                mg_dim = mg_out*num_heads
            else:
                mg_dim = mg_out

        self.drop = nn.Dropout(p=0.5)

        '''
        最终的分类器
        '''
        # self.mg_classifier2 = nn.Linear(mg_out*num_heads*2,outputs)
        self.mg_classifier = nn.Linear(mg_out*num_heads,outputs)
    
    def forward(self,mgraph,subgraph_list,na_feature,event_feature,entity_features,event_freq_feature,entity_graph_list,entity_graph_features_list,location_embed_matrix,oriented_path_list,device='cuda'):
        
        
        # ------------------------------------------------------------------------------------- #
        '''
        对子图进行学习
        subgraph_list是子图的列表
        subgraph_features是对文本学习的特点
        location_embed_matrix是地点的嵌入，将地点的嵌入和地点相关的文本的嵌入如何进来
        '''
        
        # 人名和地点之前的词的重要性
        # oriented_word_list = [[xx[1] for xx in x] for x in oriented_path_list]
        oriented_word_list = []
        for line in oriented_path_list:
            one_line = []
            for li in line:
                if li[1] in one_line:
                    continue
                else:
                    one_line.append(li[1])
            oriented_word_list.append(one_line)
        # print(oriented_word_list)
        subgraph_features_list = []
        for subg,widx in zip(subgraph_list,oriented_word_list):
            one_subg = subg.to(device)
            feats = (subg.ndata['attr']).to(device)
            if len(widx)!=0:
                word_feats = feats[widx]
                word_feats,word_feats_attn = self.sub_graph_oriented_aggregator(word_feats)
                word_feats = F.relu(self.sub_graph_oriented_linear_layer(word_feats))
                word_feats = word_feats.unsqueeze(0)
            else:
                word_feats = torch.zeros((1,self.sg_out_dim))
                word_feats = word_feats.to(device)
            for i,sla in enumerate(self.subg_list):
                feats = F.relu(sla(one_subg,feats))
            feats = self.sg_pooling(one_subg,feats)

            feats=  torch.cat([feats,word_feats],dim=1)
            feats = F.relu(self.sg_word_feats_linear_layer(feats))
            feats = feats+word_feats
            # feats = self.sg_linear(feats)
            
            subgraph_features_list.append(feats)
        subgraph_features = torch.stack(subgraph_features_list)

        
        ##################################################################
        # 融合静态地点的矩阵和动态的矩阵
        # location_embed_matrix = location_embed_matrix.unsqueeze(1)
        # location_embed_matrix = F.relu(location_embed_matrix)
        # subgraph_features = torch.cat([subgraph_features,location_embed_matrix.float()],dim=2)
        # subgraph_features = F.relu(self.sg_linear2(subgraph_features))
        ##################################################################

        # ------------------------------------------------------------------------------------- #

        # -------------------------------------------------------------------#
        '''
        人名特征的学习
        '''
        # print(na_feature)
        na_feat = self.na_emb(na_feature.to(device))
        na_feat = F.relu(self.na_layer(na_feat))
        print('na feat : ',na_feat.size())
        # na_feat = F.relu(self.na_layer((na_feature.to(device)).float()))
        # na_feat = na_feat.unsqueeze(0)
        # -------------------------------------------------------------------#
        
        # -------------------------------------------------------------------#
        '''
        事件特征的学习
        '''
        event_result = []
        event_sentence_weight = []
        if len(event_feature)==0:
            eve_feat = []
        else:
            for eve,eve_freq in zip(event_feature,event_freq_feature):
                # eve = eve.unsqueeze(0)
                eve = eve.to(device)
                eve = eve[:15]
#                 event_feat_res,attn_weight = self.event_sentence_layer(eve.float())
#                 print('event features: ',event_feat_res.size())
#                 print('attention : ',attn_weight.size())
                
                eve_feat,sentence_weight = self.ea_layer(eve.float())
                eve_feat = eve_feat.unsqueeze(0)
                sentence_weight = sentence_weight.flatten()
                event_sentence_weight.append(sentence_weight)
                eve_feat = self.ea_linear_layer(eve_feat)
                eve_feat = F.relu(eve_feat)

                #######################################################
                '''
                需要用到freq的话就加这段
                '''
                eve_freq = eve_freq.to(device)
                eve_freq = eve_freq.unsqueeze(0)
                eve_freq = self.ev_freq_layer(eve_freq.float())
                eve_feat = torch.cat([eve_feat,eve_freq],dim=1)
                eve_feat = F.relu(self.ev_freq_con(eve_feat))+eve_freq
                #######################################################

                event_result.append(eve_feat)
            eve_feat = torch.cat(event_result)
        # -------------------------------------------------------------------#

        # -------------------------------------------------------------------#
        '''
        实体的学习
        '''
        if len(entity_features)==0:
            en_feat = -1
        else:
            #################################################
            entity_feature = entity_features.to(device)
            en_feat = self.en_layer(entity_feature.float())
            en_feat = F.relu(en_feat)
            en_feat = self.drop(en_feat)
            #################################################

            ###################################################
            # en_feat_list = []
            # for ent,entf in zip(entity_graph_list,entity_graph_features_list):
            #     ent = ent.to(device)
            #     x_feat = torch.from_numpy(entf[0]).to(device)
            #     rel_feat = torch.from_numpy(entf[1]).to(device)
            #     x_edge_type = entf[2].to(device)
            #     x_edge_norm = entf[3].to(device)
            #     if len(x_edge_type)==0:
            #         x_feat = torch.zeros((1,self.en_dim))
            #         x_feat = x_feat.to(device)
            #         en_feat_list.append(x_feat)
            #         continue
            #     x_feat,rel_feat = self.en_compgcn(ent,x_feat,rel_feat,x_edge_type,x_edge_norm)
            #     one_ent_feat = F.relu(x_feat)
            #     en_feat_list.append(one_ent_feat[0].unsqueeze(0))
            # en_feat = torch.cat(en_feat_list)
            ###################################################

            
        # -------------------------------------------------------------------#
        
        
        # -------------------------------------------------------------------#
        '''
        节点的特征
        ''' 
        main_node_features = torch.concat(subgraph_features_list)        
        main_node_features = torch.cat([main_node_features,na_feat],dim=0)
        print('main_node_features_size : ',main_node_features.size())
        if len(eve_feat)!=0:
            main_node_features = torch.cat([main_node_features,eve_feat],dim=0)
        if type(en_feat)!=int:
            main_node_features = torch.cat([main_node_features,en_feat],dim=0)
        # -------------------------------------------------------------------#
        
        # -------------------------------------------------------------------#
        '''
        对行程语义图的特征进行学习
        '''
        for mgal in self.mg_list:
            mgraph = mgraph.to(device)
            main_node_features = mgal(mgraph,main_node_features)
            main_node_features = F.relu(main_node_features)
        trip_tup = main_node_features[:len(subgraph_list)]
        # -------------------------------------------------------------------#
    
        # -------------------------------------------------------------------#
        '''
        需要合并人名和地点元组就用这段
        '''
        # name_feature_vector = main_node_features[len(subgraph_list)]
        # name_feature_vector = name_feature_vector.expand((trip_tup.size()))
        # trip_tup = torch.cat((trip_tup,name_feature_vector),dim=1)
        # trip_tup = self.mg_classifier2(trip_tup)
        # -------------------------------------------------------------------#
        
        trip_tup = self.mg_classifier(trip_tup)
        return trip_tup,event_sentence_weight



import torch
from torch import nn
import dgl
import dgl.function as fn
import numpy as np

from torch.fft import irfft2
from torch.fft import rfft2
def rfft(x,d):
    t = rfft2(x,dim=(-d))
    return torch.stack((t.real,t.imag),-1)
def irfft(x,d,signal_sizes):
    return irfft2(torch.complex(x[:,:,0],x[:,:,1]),s=signal_sizes,dim=(-d))


class CompGCNCov(nn.Module):
    def __init__(self, in_channels, out_channels, act=lambda x: x, bias=True, drop_rate=0., opn='corr', num_base=-1,
                 num_rel=None):
        super(CompGCNCov, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act  # activation function
        self.device = None
        self.rel = None
        self.opn = opn

        # relation-type specific parameter
        self.in_w = self.get_param([in_channels, out_channels])
        self.out_w = self.get_param([in_channels, out_channels])
        self.loop_w = self.get_param([in_channels, out_channels])
        self.w_rel = self.get_param([in_channels, out_channels])  # transform embedding of relations to next layer
        self.loop_rel = self.get_param([1, in_channels])  # self-loop embedding

        self.drop = nn.Dropout(drop_rate)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        if num_base > 0:
            self.rel_wt = self.get_param([num_rel * 2, num_base])
        else:
            self.rel_wt = None

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def message_func(self, edges: dgl.udf.EdgeBatch):
        edge_type = edges.data['type']  # [E, 1]
        edge_num = edge_type.shape[0]
        # print('edge _type ',type(edge_type))
        # print('edge tpye ',edge_type)
        # print(self.rel[edge_type].size())
        edge_data = self.comp(edges.src['h'], self.rel[edge_type])  # [E, in_channel]
        # msg = torch.bmm(edge_data.unsqueeze(1),
        #                 self.w[edge_dir.squeeze()]).squeeze()  # [E, 1, in_c] @ [E, in_c, out_c]
        # msg = torch.bmm(edge_data.unsqueeze(1),
        #                 self.w.index_select(0, edge_dir.squeeze())).squeeze()  # [E, 1, in_c] @ [E, in_c, out_c]
        # NOTE: first half edges are all in-directions, last half edges are out-directions.
        msg = torch.cat([torch.matmul(edge_data[:edge_num // 2, :], self.in_w),
                         torch.matmul(edge_data[edge_num // 2:, :], self.out_w)])
        msg = msg * edges.data['norm'].reshape(-1, 1)  # [E, D] * [E, 1]
        return {'msg': msg}

    def reduce_func(self, nodes: dgl.udf.NodeBatch):
        return {'h': self.drop(nodes.data['h']) / 3}

    def comp(self, h, edge_data):
        def com_mult(a, b):
            r1, i1 = a[..., 0], a[..., 1]
            r2, i2 = b[..., 0], b[..., 1]
            return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)

        def conj(a):
            a[..., 1] = -a[..., 1]
            return a

        def ccorr(a, b):
            return irfft(com_mult(conj(rfft(a, 1)), rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))

        if self.opn == 'mult':
            return h * edge_data
        elif self.opn == 'sub':
            return h - edge_data
        elif self.opn == 'corr':
            return ccorr(h, edge_data.expand_as(h))
        else:
            raise KeyError(f'composition operator {self.opn} not recognized.')

    def forward(self, g: dgl.DGLGraph, x, rel_repr, edge_type, edge_norm):
        """
        :param g: dgl Graph, a graph without self-loop
        :param x: input node features, [V, in_channel]
        :param rel_repr: input relation features: 1. not using bases: [num_rel*2, in_channel]
                                                  2. using bases: [num_base, in_channel]
        :param edge_type: edge type, [E]
        :param edge_norm: edge normalization, [E]
        :return: x: output node features: [V, out_channel]
                 rel: output relation features: [num_rel*2, out_channel]
        """
        self.device = x.device
        g = g.local_var()
        g.ndata['h'] = x
        g.edata['type'] = edge_type
        g.edata['norm'] = edge_norm
        if self.rel_wt is None:
            self.rel = rel_repr
        else:
            self.rel = torch.mm(self.rel_wt, rel_repr)  # [num_rel*2, num_base] @ [num_base, in_c]
        g.update_all(self.message_func, fn.sum(msg='msg', out='h'), self.reduce_func)
        x = g.ndata.pop('h') + torch.mm(self.comp(x, self.loop_rel), self.loop_w) / 3
        if self.bias is not None:
            x = x + self.bias
        x = self.bn(x)
        # 边只更新一次，集
        return self.act(x), torch.matmul(self.rel, self.w_rel)