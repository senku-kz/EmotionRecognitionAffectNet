
db_sql_file = '../dataset/sqlite3.db'
dataset_lst = ['train_set', 'val_set']
folder_dst_lst = ['../dataset/train', '../dataset/validation']
# folder_dst_dic = {k: v for k, v in zip(dataset_lst, folder_dst_lst)}
folder_dst_dic = {i.split('/')[-1]: i for i in folder_dst_lst}

# db_sql_file = '../dataset/sqlite3.db'
# dataset_lst = ['val_set']
# folder_dst_lst = ['../dataset/validation']
# folder_dst_dic = {i.split('/')[-1]: i for i in folder_dst_lst}

# for train models
batch_size = 8
epoch_number = 3
learning_rate = 0.001

# model_dst = '../dataset/model_trained'
model_dst = './model_trained'
