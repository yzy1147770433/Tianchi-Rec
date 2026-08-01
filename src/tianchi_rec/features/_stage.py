import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import gc, os
import logging
import time
from pathlib import Path
from tianchi_rec.config import DATA_DIR, OFFLINE_DIR, ONLINE_DIR, env_path
from tianchi_rec.features import builder as feature_builder
from tianchi_rec.features import candidates as candidate_ops
from tianchi_rec.features import data as feature_data
from tianchi_rec.features import user as user_features

try:
    import lightgbm as lgb
except ModuleNotFoundError:
    lgb = None

try:
    from gensim.models import Word2Vec
except ModuleNotFoundError:
    Word2Vec = None

from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
# 节省内存的一个函数
# 减少内存
def reduce_mem(df):
    starttime = time.time()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isnull(c_min) or pd.isnull(c_max):
                continue
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem,
                                                                                                           100*(start_mem-end_mem)/start_mem,
                                                                                                           (time.time()-starttime)/60))
    return df

OFFLINE = os.environ.get('OFFLINE_VALIDATION', '1') == '1'
VALID_USER_NUMS = int(os.environ.get('VALID_USER_NUMS', '20000'))
np.random.seed(42)

data_path = str(DATA_DIR) + os.sep
default_result_dir = OFFLINE_DIR if OFFLINE else ONLINE_DIR
result_dir = env_path('RECALL_RESULT_DIR', default_result_dir)
result_dir.mkdir(parents=True, exist_ok=True)
save_path = str(result_dir) + os.sep
RECALL_METHOD = os.environ.get('RECALL_METHOD', 'itemcf')
USE_MULTI_RECALL = RECALL_METHOD == 'multi'
SINGLE_RECALL_MODEL = 'i2i_itemcf'
FORCE_REBUILD_FEATURES = os.environ.get(
    'FORCE_REBUILD_FEATURES',
    '1' if OFFLINE else '0',
) == '1'

# all_click_df指的是训练集
# sample_user_nums 采样作为验证集的用户数量

# 获取当前数据的历史点击和最后一次点击



# 返回多路召回列表或者单路召回
    


# 可以通过字典查询对应的item的Embedding

def get_article_info_df():
    article_info_df = pd.read_csv(data_path + 'articles.csv')
    article_info_df = reduce_mem(article_info_df)
    
    return article_info_df

# 这里offline的online的区别就是验证集是否为空
click_trn, click_val, click_tst, val_ans = feature_data.load_click_splits(
    DATA_DIR,
    offline=OFFLINE,
    valid_user_count=VALID_USER_NUMS,
)

click_trn_hist, click_trn_last = feature_data.get_hist_and_last_click(click_trn)

if click_val is not None:
    click_val_hist, click_val_last = click_val, val_ans
else:
    click_val_hist, click_val_last = None, None
    
click_tst_hist = click_tst

# 将召回列表转换成df的形式

# 负采样函数，这里可以控制负采样时的比例, 这里给了一个默认的值

# 召回数据打标签


# 读取 Recall.py 生成的召回结果；默认沿用原脚本的 ItemCF，想使用多路召回时把 USE_MULTI_RECALL 改成 True。
recall_list_dict = feature_data.load_recall_results(result_dir, RECALL_METHOD)
missing_tst_users = set(click_tst_hist['user_id'].unique()) - set(recall_list_dict.keys())
if missing_tst_users:
    print(f'提示：当前召回结果缺少 {len(missing_tst_users)} 个测试用户。'
          f'这通常说明 Recall.py 是 metric_recall=True 的线下模式，本次将只生成已有召回用户的特征。')
# 将召回数据转换成df
recall_list_df = candidate_ops.recall_dict_to_frame(recall_list_dict)

# 给训练验证数据打标签，并负采样（这一部分时间比较久）
trn_user_item_label_df, val_user_item_label_df, tst_user_item_label_df = candidate_ops.build_labeled_candidates(
    recall_list_df,
    click_trn_hist,
    click_val_hist,
    click_tst_hist,
    click_trn_last,
    click_val_last,
)

trn_user_item_label_df.label

# 将最终的召回的df数据转换成字典的形式做排序特征


trn_user_item_label_tuples_dict = candidate_ops.labeled_frame_to_dict(trn_user_item_label_df)

if val_user_item_label_df is not None:
    val_user_item_label_tuples_dict = candidate_ops.labeled_frame_to_dict(val_user_item_label_df)
else:
    val_user_item_label_tuples_dict = None
    
tst_user_item_label_tuples_dict = candidate_ops.labeled_frame_to_dict(tst_user_item_label_df)

# 下面基于data做历史相关的特征

article_info_df = get_article_info_df()

trn_feats_path = save_path + 'trn_user_item_feats_df.csv'
val_feats_path = save_path + 'val_user_item_feats_df.csv'
tst_feats_path = save_path + 'tst_user_item_feats_df.csv'

if not FORCE_REBUILD_FEATURES and os.path.exists(trn_feats_path) and os.path.exists(tst_feats_path):
    print('复用已生成的基础排序特征 CSV。')
else:
    if OFFLINE:
        history_frames = [click_trn_hist]
        if click_val_hist is not None:
            history_frames.append(click_val_hist)
        all_click = pd.concat(history_frames, ignore_index=True)
    else:
        all_click = pd.concat([click_trn, click_tst], ignore_index=True)
    item_content_emb_dict, item_w2v_emb_dict, item_youtube_emb_dict, user_youtube_emb_dict = feature_data.load_embedding_caches(result_dir)

    # 获取训练验证及测试数据中召回列文章相关特征
    trn_user_item_feats_df = feature_builder.create_candidate_features(
        trn_user_item_label_tuples_dict.keys(), trn_user_item_label_tuples_dict,
        click_trn_hist, article_info_df, item_content_emb_dict,
    )

    if val_user_item_label_tuples_dict is not None:
        val_user_item_feats_df = feature_builder.create_candidate_features(
            val_user_item_label_tuples_dict.keys(), val_user_item_label_tuples_dict,
            click_val_hist, article_info_df, item_content_emb_dict,
        )
    else:
        val_user_item_feats_df = None
        
    tst_user_item_feats_df = feature_builder.create_candidate_features(
        tst_user_item_label_tuples_dict.keys(), tst_user_item_label_tuples_dict,
        click_tst_hist, article_info_df, item_content_emb_dict,
    )

    # 保存一份省的每次都要重新跑，每次跑的时间都比较长
    trn_user_item_feats_df.to_csv(trn_feats_path, index=False)

    if val_user_item_feats_df is not None:
        val_user_item_feats_df.to_csv(val_feats_path, index=False)

    tst_user_item_feats_df.to_csv(tst_feats_path, index=False)    

click_tst.head()

# 读取文章特征
articles =  pd.read_csv(data_path+'articles.csv')
articles = reduce_mem(articles)

# 日志数据，就是前面的所有数据
if OFFLINE:
    history_frames = [click_trn_hist]
    if click_val_hist is not None:
        history_frames.append(click_val_hist)
    all_data = pd.concat(history_frames, ignore_index=True)
else:
    all_data = pd.concat([click_trn, click_tst], ignore_index=True)
all_data = reduce_mem(all_data)

# 拼上文章信息
all_data = all_data.merge(articles, left_on='click_article_id', right_on='article_id')

all_data.shape


user_act_fea = user_features.active_level(all_data)

user_act_fea.head()


article_hot_fea = user_features.hot_level(all_data)

article_hot_fea.head()


# 设备特征(这里时间会比较长)
device_cols = ['user_id', 'click_environment', 'click_deviceGroup', 'click_os', 'click_country', 'click_region', 'click_referrer_type']
user_device_info = user_features.device_features(all_data, device_cols)

user_device_info.head()


user_time_hob_cols = ['user_id', 'click_timestamp', 'created_at_ts']
user_time_hob_info = user_features.time_preference_features(all_data)


user_category_hob_cols = ['user_id', 'category_id']
user_cat_hob_info = user_features.category_preference_features(all_data)

user_wcou_info = all_data.groupby('user_id')['words_count'].agg('mean').reset_index()
user_wcou_info.rename(columns={'words_count': 'words_hbo'}, inplace=True)

# 所有表进行合并
user_info = pd.merge(user_act_fea, user_device_info, on='user_id')
user_info = user_info.merge(user_time_hob_info, on='user_id')
user_info = user_info.merge(user_cat_hob_info, on='user_id')
user_info = user_info.merge(user_wcou_info, on='user_id')

# 这样用户特征以后就可以直接读取了
user_info.to_csv(save_path + 'user_info.csv', index=False)   

# 本次运行继续使用内存中的 user_info，避免 cate_list 写入 CSV 后变成字符串导致 is_cat_hab 计算错误。

if os.path.exists(save_path + 'trn_user_item_feats_df.csv'):
    trn_user_item_feats_df = pd.read_csv(save_path + 'trn_user_item_feats_df.csv')
    
if os.path.exists(save_path + 'tst_user_item_feats_df.csv'):
    tst_user_item_feats_df = pd.read_csv(save_path + 'tst_user_item_feats_df.csv')

if OFFLINE and os.path.exists(save_path + 'val_user_item_feats_df.csv'):
    val_user_item_feats_df = pd.read_csv(save_path + 'val_user_item_feats_df.csv')
else:
    val_user_item_feats_df = None


# 拼上用户特征
# 下面是线下验证的
trn_user_item_feats_df = trn_user_item_feats_df.merge(user_info, on='user_id', how='left')

if val_user_item_feats_df is not None:
    val_user_item_feats_df = val_user_item_feats_df.merge(user_info, on='user_id', how='left')
else:
    val_user_item_feats_df = None
    
tst_user_item_feats_df = tst_user_item_feats_df.merge(user_info, on='user_id',how='left')

trn_user_item_feats_df.columns


articles =  pd.read_csv(data_path+'articles.csv')
articles = reduce_mem(articles)

# 拼上文章特征
trn_user_item_feats_df = trn_user_item_feats_df.merge(articles, left_on='click_article_id', right_on='article_id')

if val_user_item_feats_df is not None:
    val_user_item_feats_df = val_user_item_feats_df.merge(articles, left_on='click_article_id', right_on='article_id')
else:
    val_user_item_feats_df = None

tst_user_item_feats_df = tst_user_item_feats_df.merge(articles, left_on='click_article_id', right_on='article_id')


trn_user_item_feats_df = user_features.add_category_preference(trn_user_item_feats_df)
val_user_item_feats_df = user_features.add_category_preference(val_user_item_feats_df)
tst_user_item_feats_df = user_features.add_category_preference(tst_user_item_feats_df)

# 线下验证
del trn_user_item_feats_df['cate_list']

if val_user_item_feats_df is not None:
    del val_user_item_feats_df['cate_list']
else:
    val_user_item_feats_df = None
    
del tst_user_item_feats_df['cate_list']

del trn_user_item_feats_df['article_id']

if val_user_item_feats_df is not None:
    del val_user_item_feats_df['article_id']
else:
    val_user_item_feats_df = None
    
del tst_user_item_feats_df['article_id']

# 训练验证特征
trn_user_item_feats_df.to_csv(save_path + 'trn_user_item_feats_df.csv', index=False)
if val_user_item_feats_df is not None:
    val_user_item_feats_df.to_csv(save_path + 'val_user_item_feats_df.csv', index=False)
tst_user_item_feats_df.to_csv(save_path + 'tst_user_item_feats_df.csv', index=False)
