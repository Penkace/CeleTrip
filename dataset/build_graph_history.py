import os
from random import shuffle
import pandas as pd
import numpy as np
import math


import datetime
# 保存结果
import pickle

def save_pkl(path,obj):
    with open(path, 'wb') as f:
        pickle.dump(obj,f)
        
def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stopwords_list = list(set(stopwords.words('english')))
from nltk import sent_tokenize

import gensim
# 句子的表示利用词向量的和
# 首先对所有的预料做训练
import re
from nltk.stem.snowball import SnowballStemmer
def review_to_wordlist(review):
    words = review.lower().split()
    words = [w for w in words if not w in stopwords_list]
    
    review_text = " ".join(words)

    # Clean the text
    review_text = re.sub(r"[^A-Za-z0-9(),!.?\'\`]", " ", review_text)
    review_text = re.sub(r"\'s", " 's ", review_text)
    review_text = re.sub(r"\'ve", " 've ", review_text)
    review_text = re.sub(r"n\'t", " 't ", review_text)
    review_text = re.sub(r"\'re", " 're ", review_text)
    review_text = re.sub(r"\'d", " 'd ", review_text)
    review_text = re.sub(r"\'ll", " 'll ", review_text)
    review_text = re.sub(r",", " ", review_text)
    review_text = re.sub(r"\.", " ", review_text)
    review_text = re.sub(r"!", " ", review_text)
    review_text = re.sub(r"\(", " ( ", review_text)
    review_text = re.sub(r"\)", " ) ", review_text)
    review_text = re.sub(r"\?", " ", review_text)
    review_text = re.sub(r"\s{2,}", " ", review_text)
    
    emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                           "]+", flags=re.UNICODE)
    review_text = emoji_pattern.sub(r'', review_text)
    # url
    url = re.compile(r'https?://\S+|www\.\S+')
    review_text = url.sub(r'',review_text)
    # html
    html=re.compile(r'<.*?>')
    review_text = html.sub(r'',review_text)
    
    
    words = review_text.split()
    
    # Shorten words to their stems
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in words]
    
    review_text = " ".join(stemmed_words)
    return(review_text)


    
from collections import defaultdict
from math import log

def build_text_graph(doc_list, window_size):
    new_doc_list = []
    for doc in doc_list:
        one_doc = review_to_wordlist(doc)
        new_doc_list.append(one_doc)
    doc_list = new_doc_list
    word_freq = get_vocab(doc_list)
    vocab = list(word_freq.keys())
    word_id_map = {word: i for i, word in enumerate(vocab)}
    id_word_map = {i:word for i,word in enumerate(vocab)}
    row,col,weight = build_edges(doc_list, word_id_map, vocab, window_size)
    return row,col,weight,word_id_map,id_word_map,word_freq


def build_edges(doc_list, word_id_map,id_word_map, vocab, date_word_pair_pim_dict,window_size=10):

    windows = []
    for doc_words in doc_list:
        words = doc_words.split()
        doc_length = len(words)
        if doc_length <= window_size:
            windows.append(words)
        else:
            for i in range(doc_length - window_size + 1):
                window = words[i: i + window_size]
                windows.append(window)
    # constructing all single word frequency
    word_window_freq = defaultdict(int)
    for window in windows:
        appeared = set()
        for word in window:
            if word not in appeared:
                word_window_freq[word] += 1
                appeared.add(word)
    # constructing word pair count frequency
    word_pair_count = defaultdict(int)
    for window in windows:
        for i in range(1, len(window)):
            for j in range(i):
                word_i = window[i]
                word_j = window[j]
                word_i_id = word_id_map[word_i]
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_count[(word_i_id, word_j_id)] += 1
                word_pair_count[(word_j_id, word_i_id)] += 1
    row = []
    col = []
    weight = []
    
    # pmi as weights
    num_window = len(windows)
    for word_id_pair, count in word_pair_count.items():
        i, j = word_id_pair[0], word_id_pair[1]
#         word_freq_i = word_window_freq[vocab[i]]
#         word_freq_j = word_window_freq[vocab[j]]
        
        word_freq_i = word_window_freq[id_word_map[i]]
        word_freq_j = word_window_freq[id_word_map[j]]

        # 前面是找到当前文档的词共现
        # pmi = log((1.0 * count / num_window) /
        #           (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
        if (id_word_map[i],id_word_map[j]) in date_word_pair_pim_dict:
            pmi = date_word_pair_pim_dict[(id_word_map[i],id_word_map[j])]
        else:
            pmi = 1

        if pmi <= 0:
            continue
        row.append(i)
        col.append(j)
        weight.append(pmi)
        
    if len(row)==0:
        for kw in word_id_map:
            row.append(word_id_map[kw])
            col.append(word_id_map[kw])
            weight.append(1)
    number_nodes = len(vocab)
    number_nodes2 = np.array(row).max()
#     print('vocab list : ',number_nodes,', row list : ',number_nodes2)
    return row,col,weight,number_nodes


def get_vocab(text_list):
    word_freq = defaultdict(int)
    for doc_words in text_list:
        words = doc_words.split()
        for word in words:
            word_freq[word] += 1
    return word_freq



def construct_graph(src_list,dst_list,weight_list,graph_id,save_path):
    graph_structure=[]
    for i,(src,dst,we) in enumerate(zip(src_list,dst_list,weight_list)):
        graph_structure.append([graph_id,src,dst,we])
    one_graph = pd.DataFrame(graph_structure,columns=['graph_id','src','dst','weight'])
    one_graph.to_csv(save_path+str(graph_id)+'.csv',index=False,encoding='utf-8')

def save_oriented_path(path_list,graph_id,save_path):
    df = pd.DataFrame(path_list,columns=['name','mid','loc'])
    df.to_csv(save_path+str(graph_id)+'.csv',index=False,encoding='utf-8')   


'''
构建图的框架
group表示传入的数据模式
related_event_sentence表示与事件相关的句子
config保存的路径的类
max_length表示事件的句子的数量
word_embedding_dim表示词嵌入的框架
'''
def generate_graph(data,related_event_sentence,config,max_length = 50,word_embedding_dim=100,is_Train=True):
    group = data.groupby(['name','date'])
    group = dict(list(group))
    
    cnt = 0
    main_cnt = 0
    properties_list = []
    pmi_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/PMI_Dict/'
    for k in group:
#         if main_cnt!=169:
#             main_cnt+=1
#             continue
        date_is_ = k[1]
        date_word_pair_pim_dict = load_pkl(pmi_path+date_is_+'.pkl')
        '''
        保存主图的节点和边以及起那种
        '''
        row = []
        col = []
        main_weight = []
        one_df = group[k]
        one_df['label'] = one_df['label'].map(int)
        label_list = one_df['label'].tolist()
        name_list = one_df['name'].tolist()
        date_list = one_df['date'].tolist()
        sen_list = one_df['sen_list'].tolist()
        entity_list = one_df['entity'].tolist()
        event_list = one_df['event list'].tolist()
        loc_list = one_df['location'].tolist()

        
        
        #--------------------- 地点的索引 ---------------------#
        node_dict = {}
        node_idx = 0
        start_loc = node_idx
        for i,loc in enumerate(loc_list):
            node_dict[node_idx] = loc # 索引——>地点
            node_idx+=1
        end_loc = node_idx
        #------------------------------------------------------#

        
        
        #--------------------- 人名index ---------------------#
        '''
        从Entity中提取出这些文本中存在的人名
        '''
        this_date_name_list = []
        this_date_name_list.append(name_list[0])
        new_entity_list = []
        for i,line in enumerate(entity_list):
            name_remove_list = []
            for j,ent in enumerate(line):
                for pna in person_history_list:
                    if ent in pna or pna in ent:
                        name_remove_list.append(ent)
                        if pna not in this_date_name_list:
                            this_date_name_list.append(pna)
            new_line = line[:]      
            for nrnl in name_remove_list:
                if nrnl in new_line:
                    new_line.remove(nrnl)
            new_entity_list.append(new_line)
            
        entity_list = new_entity_list[:]
        start_name = node_idx
        this_date_name_dict = {}
        for i,nal in enumerate(this_date_name_list):
            if nal in this_date_name_dict:
                continue
            name_split_list = nal.split()
            name_split_list = [x for x in name_split_list if len(x)>1]
            this_date_name_dict[nal] = node_idx
            node_dict[node_idx] = nal
            node_idx+=1
            if i==0:
                for j in range(end_loc):
                    row.append(this_date_name_dict[nal])
                    col.append(j)
                    row.append(j)
                    col.append(this_date_name_dict[nal])
                    main_weight.append(1)
                    main_weight.append(1)
            else:
                for j in range(end_loc):
                    this_loc_sen = sen_list[j]
                    this_loc_sen = ' '.join(this_loc_sen)
                    name_exist_flag = 0
                    for na in name_split_list:
                        if na in this_loc_sen:
                            name_exist_flag = 1
                            break
                    if name_exist_flag==1:
                        row.append(this_date_name_dict[nal])
                        col.append(j)
                        row.append(j)
                        col.append(this_date_name_dict[nal])
                        main_weight.append(1)
                        main_weight.append(1)
        end_name = node_idx
        #----------------------------------------------------#
        

        #-------------------- 事件index ---------------------#
        event_dict  ={}
        event_sen = {}
        start_event = node_idx
        for i,line in enumerate(event_list):
            if len(line)==0:
                continue
            
            for eve in line:
                # 需要判断这个是否在event_article_dict中
                tup = (name_list[0],date_list[0],eve)
                if tup not in related_event_sentence:
                    continue
                
                eve_sen = related_event_sentence[tup]
                eve_sen = [x for x in eve_sen if eve in x]# 提取包含事件的句子
                
                if len(eve_sen)==0:
                    continue
                
                if tup not in event_dict:# 这个事件没有出现过
                    event_sen[tup] = eve_sen
                    event_dict[tup] = node_idx
                    node_dict[node_idx] = tup# 索引——>事件名
                    node_idx+=1
                    
                    # row.append(i)
                    # col.append(event_dict[tup])
                    # row.append(event_dict[tup])
                    # col.append(i)
                    
                    # main_weight.append(1)
                    # main_weight.append(1)
                    
                    # 如果事件还出现在其他的文章文章句子
                    for j,sen in enumerate(sen_list):
                        # if i==j:
                        #     continue
                        this_sen_str = ' '.join(sen)
                        if eve in this_sen_str:
                            row.append(event_dict[tup])
                            col.append(j)
                            row.append(j)
                            col.append(event_dict[tup])
                            main_weight.append(1)
                            main_weight.append(1)
                    
                    # ---------------人物是否出现在这些事件的句子中 ------------- #
                    one_event_sens_str = related_event_sentence[tup]
                    one_event_sens_str = ' '.join(one_event_sens_str)# 这一天和事件相关的句子
                    for tna in this_date_name_dict:
                        name_split_list = tna.split()
                        name_split_list = [x.strip() for x in name_split_list if len(x)>1]
                        name_flag = 0
                        for na in name_split_list:
                            if na in one_event_sens_str:
                                name_flag = 1
                                break
                        if name_flag==1:
                            row.append(this_date_name_dict[tna])
                            col.append(event_dict[tup])
                            row.append(event_dict[tup])
                            col.append(this_date_name_dict[tna])
                            main_weight.append(1)
                            main_weight.append(1)
                    # ---------------人物是否出现在这些事件的句子中 ------------- #
        end_event = node_idx
        #-------------------- 事件index ---------------------#
        
        
        #--------------------------------Entity List--------------------------------#
        entity_dict = {}
        ent_list = []
        start_entity = node_idx
        for i,line in enumerate(entity_list):
            if len(line)==0:
                continue
            remove_list = []
            for j,ent in enumerate(line):
                if ent not in ent_rel_dict:
                    remove_list.append(ent)
                    continue
                for k,ent2 in enumerate(line):
                    if j==k:
                        continue
                    if len(ent2)<len(ent) and (ent.endswith(ent2) or ent.startswith(ent2)):
                        remove_list.append(ent2)
            for ent in remove_list:
                if ent not in line:
                    continue
                line.remove(ent)
            
            for ent in line:

                #-------------------- 判断这个实体是否在Wikidata中 -------------------- #
                if ent not in ent_rel_dict:
                    continue
                #-------------------- 判断这个实体是否在Wikidata中 -------------------- #
                
                
                # ------------------- 判断这个实体是否指向人名 ------------------------ #
                exist_person_name = 0
                for na in name_split_list:
                    if len(na)>2:
                        if na in ent:
                            exist_person_name=1
                            break
                if exist_person_name==1:
                    continue
                # ------------------- 判断这个实体是否指向人名 ------------------------ #
                
                
                # ------------------- 判断这个实体是否指向已有的地名 ------------------ #
                exist_location_name = 0
                for loc in loc_list:
                    loc_split = loc.split()
                    if ent in loc_split:
                        exist_location_name = 1
                        break
                if exist_location_name==1:
                    continue
                # ------------------- 判断这个实体是否指向已有的地名 ------------------ #
                
                
                # ------------------- 计算实体到事件和地点的连边 -------------------- #
                if ent not in entity_dict:
                    entity_dict[ent] = node_idx
                    ent_list.append(ent)
                    node_dict[node_idx] = ent # 索引 ----> 实体
                    node_idx+=1
                    row.append(i)
                    col.append(entity_dict[ent])
                    row.append(entity_dict[ent])
                    col.append(i)
                    this_loc_sen = sen_list[i]
                    ent_freq = [x for x in this_loc_sen if ent in x]
                    ent_freq = len(ent_freq) / len(this_loc_sen)
                    main_weight.append(ent_freq)
                    main_weight.append(ent_freq)
                    
                    for j,line2 in enumerate(entity_list):
                        if i==j:
                            continue
                        if ent in line2:# 说明在其他的地点的文本中也检测到了这个实体
                            row.append(entity_dict[ent])
                            col.append(j)
                            row.append(j)
                            col.append(entity_dict[ent])
                            main_weight.append(1)
                            main_weight.append(1)
                # ------------------- 计算实体和事件还有地点的连边 -------------------- #
        end_entity = node_idx
        #--------------------------------Entity List--------------------------------#

        
        # --------------------------------------------------------------------- #
        '''
        建立行程事件子图的词
        需要记录每一天的子图index
        date_graph_list和地点index是对应的
        '''
        date_graph_list = []
        sub_graph_properties = []

        for i,(sen,loc) in enumerate(zip(sen_list,loc_list)):# sen_list这里就有对应的人物和地点
            word_sen_list = [review_to_wordlist(x) for x in sen]
            word_sen_list = [x.split() for x in word_sen_list]
            word_sen_list = [' '.join([x for x in line if x in word_vocab]) for line in word_sen_list]
            word_freq = get_vocab(word_sen_list)
            vocab = list(word_freq.keys())
            word_id_map = {word:j for j,word in enumerate(vocab)}
            id_word_map = {j:word for j,word in enumerate(vocab)}
            
            # 取出 人名和地点以及对应的路径
            # 计算两个地点之间的差异
            name_idx_in_graph = []
            loc_idx_in_graph = []
            for word in word_id_map:
                for na in name_split_list:
                    if na.lower() in word:
                        name_idx_in_graph.append(word_id_map[word])
                for loc in loc_list:
                    if loc.lower() in word:
                        loc_idx_in_graph.append(word_id_map[word])

            
#             sub_weight,sub_row,sub_col,sub_number_nodes = build_edges(word_sen_list,vocab,word_id_map,id_word_map,window_size=15)
            sub_row,sub_col,sub_weight,sub_number_nodes= build_edges(word_sen_list, word_id_map,id_word_map, vocab, date_word_pair_pim_dict,window_size=5)
            first_idx_list = []
            for ridx,cidx in zip(sub_row,sub_col):
                if ridx in name_idx_in_graph and cidx not in name_idx_in_graph and cidx not in loc_idx_in_graph:
                    first_idx_list.append([ridx,cidx])

            first_order_path  =[]
            for tup in first_idx_list:
                for ridx,cidx in zip(sub_row,sub_col):
                    if tup[1]==ridx and cidx in loc_idx_in_graph:
                        if len(id_word_map[tup[1]])<=1:
                            continue
                        first_order_path.append([tup[0],tup[1],cidx])
            ###############
            '''
            打印路径
            '''
#             print(len(first_order_path))
#             for tup in first_order_path:
#                 print(id_word_map[tup[0]],id_word_map[tup[1]],id_word_map[tup[2]])
            ###############
                
#             print(main_cnt)
#             print('row node ',np.array(sub_row).max())
#             print('col node ',sub_col)
#             print('sub node number ',sub_number_nodes)
            
            if len(sub_row)==0:
                print(k,i)
            if is_Train==True:
                # 保存sub_graph
                construct_graph(sub_row,sub_col,sub_weight,cnt,save_path=config.train_sub_graph_save_path)
                save_oriented_path(first_order_path,cnt,save_path = config.train_sub_graph_oriented_path)
            
            else:
                construct_graph(sub_row,sub_col,sub_weight,cnt,save_path = config.test_sub_graph_save_path)
                save_oriented_path(first_order_path,cnt,save_path = config.test_sub_graph_oriented_path)
            
            sub_graph_properties.append([cnt,word_id_map,id_word_map,sub_number_nodes])
            date_graph_list.append(cnt)
            cnt+=1
        # --------------------------------------------------------------------- #
        
        
#         print('The sub graph length :',list(range(len(date_graph_list))))
#         print('The name : ',list(range(len(date_graph_list),len(date_graph_list)+1)))
#         print('The event : ',list(range(start_event,end_event)))
#         print('The entity : ',list(range(start_entity,end_entity)))
#         print('node dict : ',node_dict)

        
        # --------------------------------------------------------------------- #
        '''
        得到事件相关的句子
        eve_sen_dict是事件的句子特征矩阵
        '''
        eve_sen_dict = {}
        for i in range(start_event,end_event):
            tup = node_dict[i]
            eve_name= tup[2]
            eve_sen = related_event_sentence[tup]
            eve_sen = [x for x in eve_sen if eve_name in x]
            
            eve_sen = eve_sen[:max_length]
            eve_sen_matrix = np.zeros((max_length,word_embedding_dim))

            for j,one_sen in enumerate(eve_sen):
                one_sen = review_to_wordlist(one_sen).split()
                one_sen = [model.wv[x] for x in one_sen if x in word_vocab]
                one_sen = np.array(one_sen)
                one_sen_mean = one_sen.mean(axis=0)# 对所有词做加全求平均
                eve_sen_matrix[j,:] = one_sen_mean
            eve_sen_dict[i] = eve_sen_matrix
        # --------------------------------------------------------------------- #
        

        # --------------------------------------------------------------------- #
        '''
        获得实体的特征嵌入
        entity_features对应的就是实体节点的嵌入
        '''
        entity_features = np.zeros((len(entity_dict),word_embedding_dim))
        for i,kv in enumerate(ent_list):
            if kv.lower() not in word_vocab:
                entity_features[i,:] = np.random.random(100)
            else:
                entity_features[i,:] = model.wv[kv.lower()]
        # --------------------------------------------------------------------- #

        
        # --------------------------------------------------------------------- #
        '''
        保存到相应的文件夹下，先保存主图
        包括主图的实体特征，事件特征，图的边和节点特征
        main_cnt是日期图的编号
        label_list是标签的列表
        date_graph_list是人物行程元组的，用来从路径提取这个日期下的行程图，存的是cnt
        node_dict是人物三元组、人名、事件和实体的节点字典
        event_dict是事件的字典，对应的特征矩阵
        entity_dict是对应的实体矩阵，对应字典的实体
        '''
        if is_Train==True:
            save_pkl(config.train_entity_save_path+str(main_cnt)+'.pkl',entity_features)
            save_pkl(config.train_event_save_path+str(main_cnt)+'.pkl',eve_sen_dict)
            save_pkl(config.train_event_sent_save_path+str(main_cnt)+'.pkl',event_sen)
            
            # 主图的保存
#             print(row,col)
            construct_graph(row,col,main_weight,main_cnt,save_path=config.train_semantic_graph_path)
            sub_graph_properties = pd.DataFrame(sub_graph_properties,columns=['graph_id','word_id','id_word','graph_node_num'])
            
            # 保存地点子图的特征
            save_pkl(config.train_sub_graph_properties_path+str(main_cnt)+'.pkl',sub_graph_properties)
        else:
            save_pkl(config.test_entity_save_path+str(main_cnt)+'.pkl',entity_features)
            save_pkl(config.test_event_save_path+str(main_cnt)+'.pkl',eve_sen_dict)
            save_pkl(config.test_event_sent_save_path+str(main_cnt)+'.pkl',event_sen)        
            construct_graph(row,col,main_weight,main_cnt,save_path=config.test_semantic_graph_path)
            sub_graph_properties = pd.DataFrame(sub_graph_properties,columns=['graph_id','word_id','id_word','graph_node_num'])
            save_pkl(config.test_sub_graph_properties_path+str(main_cnt)+'.pkl',sub_graph_properties)
        properties_list.append([main_cnt,np.array(label_list),date_graph_list,node_dict,event_dict,entity_dict,this_date_name_dict,date_is_])
        # --------------------------------------------------------------------- #
        
        main_cnt+=1
#         if main_cnt>=150:
#             break
    if is_Train==True:
        # 保存semantic graph的节点特征
        properties_list = pd.DataFrame(properties_list,columns=['graph_id','label','sub_graph_id','node_dict','event_dict','entity_dict','this_date_person_dict','date'])
        save_pkl(config.train_semantic_properties,properties_list)
    else:
        properties_list = pd.DataFrame(properties_list,columns=['graph_id','label','sub_graph_id','node_dict','event_dict','entity_dict','this_date_person_dict','date'])
        save_pkl(config.test_semantic_properties,properties_list)




#-------------------------------------------------------------------------------------------#
from gensim.models import Word2Vec, KeyedVectors
import gensim

# 字的实体
model = gensim.models.Word2Vec.load('/public/home/pengkai/Itinerary_Miner/Article_contain_person_name/News/all_trip_dataset/trip_w2v_model.model')
word_vocab = model.wv.vocab
def exist_entity_name(ent):
    flag = 0
    try:
        word_vector = model.wv[ent.lower()]
    except:
        flag=1
    return flag

# 加载训练数据和测试数据
train_df  = load_pkl('/public/home/pengkai/Itinerary_Miner/CeleTrip/data/train.pkl')
test_df = load_pkl('/public/home/pengkai/Itinerary_Miner/CeleTrip/data/test.pkl')
article_dict = load_pkl('/public/home/pengkai/Itinerary_Miner/HeteroTrip/data/article_dict.pkl')
train_event_sentence = load_pkl('/public/home/pengkai/Itinerary_Miner/CeleTrip/data/train_event_related_sentence.pkl')
test_event_sentence = load_pkl('/public/home/pengkai/Itinerary_Miner/CeleTrip/data/test_event_related_sentence.pkl')

# 加载实体的结果
ent_rel_dict = load_pkl('/public/home/pengkai/Itinerary_Miner/CeleTrip/data/ent_rel_dict.pkl')
ent_features = load_pkl('/public/home/pengkai/Itinerary_Miner/HeteroTrip/data/ent_openke_feature.pkl')
rel_features = load_pkl('/public/home/pengkai/Itinerary_Miner/HeteroTrip/data/rel_openke_feature.pkl')

history_dict= load_pkl('/public/home/pengkai/Itinerary_Miner/CeleTrip/data/person_history.pkl')
person_history_list = history_dict.keys()
train_df['label'].value_counts(),test_df['label'].value_counts()


#####################################################################################
class Config:
    save_path = '/public/home/pengkai/Itinerary_Miner/CeleTrip/graph_data2/path_oriented_history_w5_text_entity_event_history/'
    
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
#####################################################################################

config = Config()
# 保存的主路径
if not os.path.exists(config.save_path):
    os.makedirs(config.save_path)

# semantic graph的保存路径
if not os.path.exists(config.train_semantic_graph_path):
    os.makedirs(config.train_semantic_graph_path)
    
if not os.path.exists(config.test_semantic_graph_path):
    os.makedirs(config.test_semantic_graph_path)

# 地点子图的边关系保存路径
if not os.path.exists(config.train_sub_graph_save_path):
    os.makedirs(config.train_sub_graph_save_path)

if not os.path.exists(config.test_sub_graph_save_path):
    os.makedirs(config.test_sub_graph_save_path)
    
# 地点子图的特征保存路径    
if not os.path.exists(config.train_sub_graph_properties_path):
    os.makedirs(config.train_sub_graph_properties_path)
    
if not os.path.exists(config.test_sub_graph_properties_path):
    os.makedirs(config.test_sub_graph_properties_path)
    
    
# 实体子图的保存路径保存
if not os.path.exists(config.train_entity_save_path):
    os.makedirs(config.train_entity_save_path)
    
if not os.path.exists(config.test_entity_save_path):
    os.makedirs(config.test_entity_save_path)

# 事件句子保存路径保存
if not os.path.exists(config.train_event_save_path):
    os.makedirs(config.train_event_save_path)
    
if not os.path.exists(config.test_event_save_path):
    os.makedirs(config.test_event_save_path)

# 事件句子的权重保存
if not os.path.exists(config.train_event_sent_save_path):
    os.makedirs(config.train_event_sent_save_path)
    
if not os.path.exists(config.test_event_sent_save_path):
    os.makedirs(config.test_event_sent_save_path)
    
if not os.path.exists(config.train_sub_graph_oriented_path):
    os.makedirs(config.train_sub_graph_oriented_path)
    
if not os.path.exists(config.test_sub_graph_oriented_path):
    os.makedirs(config.test_sub_graph_oriented_path)
#####################################################################
'''
消融的数据集构建
'''
# none_list = [[] for _ in range(len(train_df))]
# train_df['event list'] = none_list

# none_list = [[] for _ in range(len(test_df))]
# test_df['event list'] = none_list

# none_list = [[] for _ in range(len(train_df))]
# train_df['entity'] = none_list

# none_list = [[] for _ in range(len(test_df))]
# test_df['entity'] = none_list

#####################################################################



print('###### Start Train ######')
train_df.head()
generate_graph(train_df,train_event_sentence,config,max_length = 50,word_embedding_dim=100,is_Train=True)
print('###### Finish Train ######')

print('###### Start Test ######')
test_df.head()
generate_graph(test_df,test_event_sentence,config,max_length = 50,word_embedding_dim=100,is_Train=False)
print('###### Finish Test ######')