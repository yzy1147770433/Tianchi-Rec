import pandas as pd
import numpy as np
from tqdm import tqdm
import os, warnings, pickle
import collections
warnings.filterwarnings('ignore')
from pathlib import Path
from tianchi_rec.config import DATA_DIR, OFFLINE_DIR, ONLINE_DIR, env_path
from tianchi_rec.recall import cold_start as cold_start_algo
from tianchi_rec.recall import common as recall_common
from tianchi_rec.recall import content as content_algo
from tianchi_rec.recall import itemcf as itemcf_algo
from tianchi_rec.recall import usercf as usercf_algo
from tianchi_rec.recall import youtube_dnn as youtube_algo
from tianchi_rec.recall import combine_recall_results as combine_recall_channels


np.random.seed(42)

OFFLINE = os.environ.get('OFFLINE_VALIDATION', '1') == '1'
RECALL_METHOD = os.environ.get('RECALL_METHOD', 'itemcf')
if RECALL_METHOD not in {'itemcf', 'multi'}:
    raise ValueError("RECALL_METHOD must be either 'itemcf' or 'multi'.")
ITEMCF_SIM_TOPK = int(os.environ.get('ITEMCF_SIM_TOPK', '100'))
SINGLE_RECALL_TOPK = int(os.environ.get('SINGLE_RECALL_TOPK', '50'))
FINAL_RECALL_TOPK = int(os.environ.get('FINAL_RECALL_TOPK', '50'))

data_path = DATA_DIR
default_result_dir = OFFLINE_DIR if OFFLINE else ONLINE_DIR
save_path = env_path('RECALL_RESULT_DIR', default_result_dir)
save_path.mkdir(parents=True, exist_ok=True)
# 做召回评估的一个标志, 如果不进行评估就是直接使用全量数据进行召回
metric_recall = OFFLINE


# 读取数据
# debug模式： 从训练集中划出一部分数据来调试代码

# 读取点击数据，这里分成线上和线下，如果是为了获取线上提交结果应该讲测试集中的点击数据合并到总的数据中
# 如果是为了线下验证模型的有效性或者特征的有效性，可以只使用训练集

#################################################################################################
# 读取文章的基本属性
def get_item_info_df(data_path):
    item_info_df = pd.read_csv(Path(data_path) / 'articles.csv')
    
    # 为了方便与训练集中的click_article_id拼接，需要把article_id修改成click_article_id
    item_info_df = item_info_df.rename(columns={'article_id': 'click_article_id'})
    
    return item_info_df

# 读取文章的Embedding数据
def get_item_emb_dict(data_path):
    item_emb_df = pd.read_csv(Path(data_path) / 'articles_emb.csv')
    
    item_emb_cols = [x for x in item_emb_df.columns if 'emb' in x]
    item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols], dtype=np.float32)
    # 进行归一化
    item_emb_norm = np.linalg.norm(item_emb_np, axis=1, keepdims=True)
    item_emb_np = item_emb_np / np.maximum(item_emb_norm, 1e-12)

    item_emb_dict = dict(zip(item_emb_df['article_id'], item_emb_np))
    pickle.dump(item_emb_dict, open(save_path / 'item_content_emb.pkl', 'wb'))
    
    return item_emb_dict

max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
# 采样数据
# all_click_df = get_all_click_sample(data_path)

# 全量训练集
all_click_df = recall_common.load_clicks(data_path, offline=OFFLINE)

# 对时间戳进行归一化,用于在关联规则的时候计算权重
all_click_df['click_timestamp'] = all_click_df[['click_timestamp']].apply(max_min_scaler)
item_info_df = get_item_info_df(data_path)
item_content_emb_path = save_path / 'item_content_emb.pkl'
if item_content_emb_path.exists():
    print(f'复用缓存: {item_content_emb_path}')
    item_emb_dict = None  # 后续流程不直接使用该字典
else:
    item_emb_dict = get_item_emb_dict(data_path)


#################################################################################################
# 以下是工具函数
# 根据点击时间获取用户的点击文章序列   {user1: [(item1, time1), (item2, time2)..]...}

# 根据时间获取商品被点击的用户序列  {item1: [(user1, time1), (user2, time2)...]...}
# 这里的时间是用户点击当前商品的时间，好像没有直接的关系。

# 获取当前数据的历史点击和最后一次点击

# 获取文章id对应的基本属性，保存成字典的形式，方便后面召回阶段，冷启动阶段直接使用



# 获取近期点击最多的文章





#################################################################################################
# 定义多路召回字典
# 获取文章的属性信息，保存成字典的形式方便查询
item_type_dict, item_words_dict, item_created_time_dict = recall_common.item_metadata(item_info_df)

# 定义一个多路召回的字典，将各路召回的结果都保存在这个字典当中
user_multi_recall_dict =  {'itemcf_sim_itemcf_recall': {},
                           'embedding_sim_item_recall': {},
                           'youtubednn_recall': {},
                           'youtubednn_usercf_recall': {}, 
                           'cold_start_recall': {}}


# 提取最后一次点击作为召回评估，如果不需要做召回评估直接使用全量的训练集进行召回(线下验证模型)
# 如果不是召回评估，直接使用全量数据进行召回，不用将最后一次提取出来
if metric_recall:
    trn_hist_click_df, trn_last_click_df = recall_common.split_history_last(all_click_df)
else:
    trn_hist_click_df = all_click_df
    trn_last_click_df = None


#################################################################################################
# 召回效果评估函数
# 依次评估召回的前10, 20, 30, 40, 50个文章中的击中率
def metrics_recall(user_recall_items_dict, trn_last_click_df, topk=5):
    last_click_item_dict = dict(zip(trn_last_click_df['user_id'], trn_last_click_df['click_article_id']))
    user_num = len(user_recall_items_dict)
    
    for k in range(10, topk+1, 10):
        hit_num = 0
        for user, item_list in user_recall_items_dict.items():
            # 获取前k个召回的结果
            tmp_recall_items = [x[0] for x in user_recall_items_dict[user][:k]]
            if last_click_item_dict[user] in set(tmp_recall_items):
                hit_num += 1
        
        hit_rate = round(hit_num * 1.0 / user_num, 5)
        print(' topk: ', k, ' : ', 'hit_num: ', hit_num, 'hit_rate: ', hit_rate, 'user_num : ', user_num)


#################################################################################################

# 计算相似性矩阵



# ItemCF 依赖当前抽样用户，必须与本次样本同步生成，不能沿用其他样本的缓存。
i2i_sim = itemcf_algo.item_similarity(trn_hist_click_df, item_created_time_dict)
pickle.dump(i2i_sim, open(save_path / 'itemcf_i2i_sim.pkl', 'wb'))




# 由于usercf计算时候太耗费内存了，这里就不直接运行了
# 如果是采样的话，是可以运行的
# user_activate_degree_dict = get_user_activate_degree_dict(all_click_df)
# u2u_sim = usercf_sim(all_click_df, user_activate_degree_dict)


# 向量检索相似度计算
# topk指的是每个item, faiss搜索后返回最相似的topk个item

emb_i2i_sim_path = save_path / 'emb_i2i_sim.pkl'
if emb_i2i_sim_path.exists():
    print(f'复用缓存: {emb_i2i_sim_path}')
    emb_i2i_sim = None  # 在实际召回前再从缓存加载
else:
    item_emb_df = pd.read_csv(data_path / 'articles_emb.csv')
    emb_i2i_sim = content_algo.embedding_similarity(item_emb_df, topk=10)
    pickle.dump(emb_i2i_sim, open(save_path / 'emb_i2i_sim.pkl', 'wb'))
    del item_emb_df




#####################################################################################################
# 多路召回
# 2026/7/3 YoutubeDNN召回
# 获取双塔召回时的训练验证数据
# negsample指的是通过滑窗构建样本的时候，负样本的数量

# 将输入的数据进行padding，使得序列特征的长度都一致


# 由于这里需要做召回评估，所以讲训练集中的最后一次点击都提取了出来
if RECALL_METHOD == 'multi':
    if not metric_recall:
        user_multi_recall_dict['youtubednn_recall'] = youtube_algo.train_youtube_dnn_recall(
            all_click_df, save_path, topk=20,
        )
    else:
        trn_hist_click_df, trn_last_click_df = recall_common.split_history_last(all_click_df)
        user_multi_recall_dict['youtubednn_recall'] = youtube_algo.train_youtube_dnn_recall(
            trn_hist_click_df, save_path, topk=20,
        )
        metrics_recall(user_multi_recall_dict['youtubednn_recall'], trn_last_click_df, topk=20)




# 2026/7/4
# 基于商品的召回i2i

# 先进行itemcf召回, 为了召回评估，所以提取最后一次点击

if metric_recall:
    trn_hist_click_df, trn_last_click_df = recall_common.split_history_last(all_click_df)
else:
    trn_hist_click_df = all_click_df

user_recall_items_dict = collections.defaultdict(dict)
user_item_time_dict = recall_common.user_item_time(trn_hist_click_df)

i2i_sim = pickle.load(open(save_path / 'itemcf_i2i_sim.pkl', 'rb'))
emb_i2i_sim = pickle.load(open(save_path / 'emb_i2i_sim.pkl', 'rb'))

sim_item_topk = ITEMCF_SIM_TOPK
recall_item_num = SINGLE_RECALL_TOPK
item_topk_click = recall_common.top_clicked_items(trn_hist_click_df, count=50)

for user in tqdm(trn_hist_click_df['user_id'].unique()):
    user_recall_items_dict[user] = itemcf_algo.recommend_items(
        user, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num,
        item_topk_click, item_created_time_dict, emb_i2i_sim,
    )

user_multi_recall_dict['itemcf_sim_itemcf_recall'] = user_recall_items_dict
pickle.dump(user_multi_recall_dict['itemcf_sim_itemcf_recall'], open(save_path / 'itemcf_recall_dict.pkl', 'wb'))

if metric_recall:
    # 召回效果评估
    metrics_recall(user_multi_recall_dict['itemcf_sim_itemcf_recall'], trn_last_click_df, topk=recall_item_num)

if RECALL_METHOD == 'itemcf':
    print(f'ItemCF recall saved to: {save_path / "itemcf_recall_dict.pkl"}')
    raise SystemExit(0)

# 这里是为了召回评估，所以提取最后一次点击
if metric_recall:
    trn_hist_click_df, trn_last_click_df = recall_common.split_history_last(all_click_df)
else:
    trn_hist_click_df = all_click_df

user_recall_items_dict = collections.defaultdict(dict)
user_item_time_dict = recall_common.user_item_time(trn_hist_click_df)
i2i_sim = pickle.load(open(save_path / 'emb_i2i_sim.pkl','rb'))

sim_item_topk = ITEMCF_SIM_TOPK
recall_item_num = SINGLE_RECALL_TOPK

item_topk_click = recall_common.top_clicked_items(trn_hist_click_df, count=50)

for user in tqdm(trn_hist_click_df['user_id'].unique()):
    user_recall_items_dict[user] = itemcf_algo.recommend_items(
        user, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num,
        item_topk_click, item_created_time_dict, emb_i2i_sim,
    )
    
user_multi_recall_dict['embedding_sim_item_recall'] = user_recall_items_dict
pickle.dump(user_multi_recall_dict['embedding_sim_item_recall'], open(save_path / 'embedding_sim_item_recall.pkl', 'wb'))

if metric_recall:
    # 召回效果评估
    metrics_recall(user_multi_recall_dict['embedding_sim_item_recall'], trn_last_click_df, topk=recall_item_num)





# 2026/7/5
#
# 基于用户的召回 u2u2i


# 这里是为了召回评估，所以提取最后一次点击
# 由于usercf中计算user之间的相似度的过程太费内存了，全量数据这里就没有跑，跑了一个采样之后的数据
# if metric_recall:
#     trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
# else:
#     trn_hist_click_df = all_click_df
    
# user_recall_items_dict = collections.defaultdict(dict)
# user_item_time_dict = get_user_item_time(trn_hist_click_df)

# u2u_sim = pickle.load(open(save_path / 'usercf_u2u_sim.pkl', 'rb'))

# sim_user_topk = 20
# recall_item_num = 10
# item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)

# for user in tqdm(trn_hist_click_df['user_id'].unique()):
#     user_recall_items_dict[user] = user_based_recommend(user, user_item_time_dict, u2u_sim, sim_user_topk, \
#                                                         recall_item_num, item_topk_click, item_created_time_dict, emb_i2i_sim)    

# pickle.dump(user_recall_items_dict, open(save_path / 'usercf_u2u2i_recall.pkl', 'wb'))

# if metric_recall:
#     # 召回效果评估
#     metrics_recall(user_recall_items_dict, trn_last_click_df, topk=recall_item_num)


# 使用Embedding的方式获取u2u的相似性矩阵
# topk指的是每个user, faiss搜索后返回最相似的topk个user

# 读取YoutubeDNN过程中产生的user embedding, 然后使用faiss计算用户之间的相似度
# 这里需要注意，这里得到的user embedding其实并不是很好，因为YoutubeDNN中使用的是用户点击序列来训练的user embedding,
# 如果序列普遍都比较短的话，其实效果并不是很好
user_emb_dict = pickle.load(open(save_path / 'user_youtube_emb.pkl', 'rb'))
u2u_sim = usercf_algo.embedding_user_similarity(user_emb_dict, topk=10)
pickle.dump(u2u_sim, open(save_path / 'youtube_u2u_sim.pkl', 'wb'))


# 使用召回评估函数验证当前召回方式的效果
if metric_recall:
    trn_hist_click_df, trn_last_click_df = recall_common.split_history_last(all_click_df)
else:
    trn_hist_click_df = all_click_df

user_recall_items_dict = collections.defaultdict(dict)
user_item_time_dict = recall_common.user_item_time(trn_hist_click_df)
u2u_sim = pickle.load(open(save_path / 'youtube_u2u_sim.pkl', 'rb'))

sim_user_topk = 20
recall_item_num = SINGLE_RECALL_TOPK

item_topk_click = recall_common.top_clicked_items(trn_hist_click_df, count=50)
for user in tqdm(trn_hist_click_df['user_id'].unique()):
    user_recall_items_dict[user] = usercf_algo.recommend_from_users(
        user, user_item_time_dict, u2u_sim, sim_user_topk, recall_item_num,
        item_topk_click, item_created_time_dict, emb_i2i_sim,
    )
    
user_multi_recall_dict['youtubednn_usercf_recall'] = user_recall_items_dict
pickle.dump(user_multi_recall_dict['youtubednn_usercf_recall'], open(save_path / 'youtubednn_usercf_recall.pkl', 'wb'))

if metric_recall:
    # 召回效果评估
    metrics_recall(user_multi_recall_dict['youtubednn_usercf_recall'], trn_last_click_df, topk=recall_item_num)



#################################################################################################
# 冷启动问题



# 先进行itemcf召回，这里不需要做召回评估，这里只是一种策略
if metric_recall:
    trn_hist_click_df,_ = recall_common.split_history_last(
        all_click_df
    )
else:
    trn_hist_click_df = all_click_df

user_recall_items_dict = collections.defaultdict(dict)
user_item_time_dict = recall_common.user_item_time(trn_hist_click_df)
i2i_sim = pickle.load(open(save_path / 'emb_i2i_sim.pkl','rb'))

sim_item_topk = 150
recall_item_num = 100 # 稍微召回多一点文章，便于后续的规则筛选

item_topk_click = recall_common.top_clicked_items(trn_hist_click_df, count=50)
for user in tqdm(trn_hist_click_df['user_id'].unique()):
    user_recall_items_dict[user] = itemcf_algo.recommend_items(
        user, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num,
        item_topk_click, item_created_time_dict, emb_i2i_sim,
    )
pickle.dump(user_recall_items_dict, open(save_path / 'cold_start_items_raw_dict.pkl', 'wb'))


# 基于规则进行文章过滤
# 保留文章主题与用户历史浏览主题相似的文章
# 保留文章字数与用户历史浏览文章字数相差不大的文章
# 保留最后一次点击当天的文章
# 按照相似度返回最终的结果




all_click_df_ = trn_hist_click_df.copy()
all_click_df_ = all_click_df_.merge(item_info_df, how='left', on='click_article_id')
user_hist_item_typs_dict, user_hist_item_ids_dict, user_hist_item_words_dict, user_last_item_created_time_dict = recall_common.user_history_metadata(all_click_df_)
click_article_ids_set = set(trn_hist_click_df['click_article_id'])
# 需要注意的是
# 这里使用了很多规则来筛选冷启动的文章，所以前面再召回的阶段就应该尽可能的多召回一些文章，否则很容易被删掉
cold_start_user_items_dict = cold_start_algo.filter_cold_start_items(
    user_recall_items_dict, user_hist_item_typs_dict, user_hist_item_words_dict,
    user_last_item_created_time_dict, item_type_dict, item_words_dict,
    item_created_time_dict, click_article_ids_set, recall_item_num,
)
pickle.dump(cold_start_user_items_dict, open(save_path / 'cold_start_user_items_dict.pkl', 'wb'))

user_multi_recall_dict['cold_start_recall'] = cold_start_user_items_dict



#################################################################################################

# 多路召回合并

def combine_recall_results(user_multi_recall_dict, weight_dict=None, topk=25):
    print('多路召回合并...')
    final_recall_items_dict_rank = combine_recall_channels(
        user_multi_recall_dict,
        weights=weight_dict,
        topk=topk,
    )
    pickle.dump(final_recall_items_dict_rank, open(os.path.join(save_path, 'final_recall_items_dict.pkl'),'wb'))
    return final_recall_items_dict_rank



# 这里直接对多路召回的权重给了一个相同的值，其实可以根据前面召回的情况来调整参数的值
weight_dict = {'itemcf_sim_itemcf_recall': 1.0,
               'embedding_sim_item_recall': 1.0,
               'youtubednn_recall': 1.0,
               'youtubednn_usercf_recall': 1.0, 
               'cold_start_recall': 1.0}

final_recall_items_dict = combine_recall_results(
    user_multi_recall_dict,
    weight_dict=weight_dict,
    topk=FINAL_RECALL_TOPK,
)
if metric_recall:
    metrics_recall(
        final_recall_items_dict,
        trn_last_click_df,
        topk=20,
    )
