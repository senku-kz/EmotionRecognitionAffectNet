
db_sql_file = '../dataset/sqlite3.db'
dataset_lst = ['train_set', 'val_set']
folder_dst_lst = ['../dataset/train', '../dataset/validation']
# folder_dst_dic = {k: v for k, v in zip(dataset_lst, folder_dst_lst)}
folder_dst_dic = {i.split('/')[-1]: i for i in folder_dst_lst}

# db_sql_file = '../dataset/sqlite3.db'
# dataset_lst = ['val_set']
# folder_dst_lst = ['../dataset/validation']
# folder_dst_dic = {i.split('/')[-1]: i for i in folder_dst_lst}

CUDA_N = 'cuda'

# for train models
batch_size = 4
epoch_number = 1
learning_rate = 0.001

LIMIT_DATASET = 100
# LIMIT_DATASET = None

SPLIT_TRAIN_SET = 0.8

# model_dst = '../dataset/model_trained'
model_dst = './model_trained'
