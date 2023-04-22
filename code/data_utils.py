import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.utils.data as Data
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, build_input_features

# key2index = {}

def split(x):
    key2index = {'Action': 1, 'Crime': 2, 'Drama': 3, 'Thriller': 4, "Children's": 5, 'Comedy': 6, \
            'Adventure': 7, 'Musical': 8, 'Romance': 9, 'War': 10, 'Film-Noir': 11, 'Mystery': 12, \
            'Documentary': 13, 'Fantasy': 14, 'Sci-Fi': 15, 'Animation': 16, 'Western': 17, 'Horror': 18}
    key_ans = x.split('|')
    return list(map(lambda x: key2index[x], key_ans))


def pad_sequences(genres_list, maxlen):
    genres_arr = np.zeros((len(genres_list),maxlen))
    for i,sublist in enumerate(genres_list):
        sublist = sublist[:maxlen]
        for _ in range(maxlen-len(sublist)):
            sublist.append(0)
        genres_arr[i] = np.array(sublist)
    return genres_arr.astype('int')


def get_data(dataset):
    if dataset == 'ml-1m':
        key2index = {'Action': 1, 'Crime': 2, 'Drama': 3, 'Thriller': 4, "Children's": 5, 'Comedy': 6, \
            'Adventure': 7, 'Musical': 8, 'Romance': 9, 'War': 10, 'Film-Noir': 11, 'Mystery': 12, \
            'Documentary': 13, 'Fantasy': 14, 'Sci-Fi': 15, 'Animation': 16, 'Western': 17, 'Horror': 18}
        # 1.Prepare data
        unames = ["UserID", "Gender", "Age", "Occupation", "Zip-code"]
        users = pd.read_csv("../data/ml-1m/users.dat", sep="::", header=None, names=unames,encoding="ISO-8859-1",engine='python')
        mnames = ["MovieID", "Title", "Genres"]
        movies = pd.read_csv("../data/ml-1m/movies.dat", sep="::", header=None, names=mnames,encoding="ISO-8859-1",engine='python')
        rnames = ["UserID", "MovieID", "Rating", "TimeStamp"]
        ratings = pd.read_csv("../data/ml-1m/ratings.dat", sep="::", header=None, names=rnames,encoding="ISO-8859-1",engine='python')
        ratings['Rating'] = ratings['Rating'].apply(lambda x:1 if x>=4 else 0)

        data = pd.merge(ratings, movies, on='MovieID')
        data = pd.merge(data, users, on='UserID')
        data.sort_values('TimeStamp',inplace=True)
        data.reset_index(drop=True,inplace=True)
        sparse_features = ["UserID", "MovieID", "Gender", "Age", "Occupation"]

        # 2.Label Encoding for sparse features,and process sequence features
        for feat in sparse_features:
            lbe = LabelEncoder()
            data[feat] = lbe.fit_transform(data[feat])
            # print(lbe.classes_)
        # preprocess the sequence feature
        genres_list = list(map(split, data["Genres"].values))
        genres_length = np.array(list(map(len, genres_list)))
        max_len = max(genres_length)
        # genres_list = pad_sequences(genres_list, maxlen=max_len)    # Notice: padding=`post`

        # 3.count #unique features for each sparse field and generate feature config for sequence feature
        fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique()) for feat in sparse_features]
        varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size=len(key2index) + 1), maxlen=max_len)]
        linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
        dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns
        feature_index = build_input_features(linear_feature_columns + dnn_feature_columns)

        # 4.split dataset
        trainval, valtest = round(len(data)*0.7), round(len(data)*0.9)
        uid, iid = data['UserID'].values, data['MovieID'].values
        # train
        train_user_set, train_item_set, train_df = set(uid[:trainval]), set(iid[:trainval]), data.iloc[0:trainval]
        train_genres_list = list(map(split, train_df["Genres"].values))
        train_dict = {name: train_df[name] for name in sparse_features}
        train_dict["genres"] = pad_sequences(train_genres_list, maxlen=max_len)
        trainx, trainy = [train_dict[feature] for feature in feature_index], train_df['Rating'].values.reshape(-1,1)
        for i in range(len(trainx)):
            if len(trainx[i].shape) == 1:
                trainx[i] = np.expand_dims(trainx[i], axis=1)
        train_tensor_data = Data.TensorDataset(torch.from_numpy(np.concatenate(trainx, axis=-1)), torch.from_numpy(trainy))
        # val
        val_df = data.iloc[trainval:valtest]
        val_df['keep'] = val_df.apply(lambda x: (x['UserID'] in train_user_set) and (x['MovieID'] in train_item_set), axis=1)
        val_df = val_df[val_df['keep']==True]
        val_df.drop(columns=['keep'], inplace=True)
        val_df.reset_index(drop=True,inplace=True)
        val_genres_list = list(map(split, val_df["Genres"].values))
        val_dict = {name: val_df[name] for name in sparse_features}
        val_dict["genres"] = pad_sequences(val_genres_list, maxlen=max_len)
        valx, valy = [val_dict[feature] for feature in feature_index], val_df['Rating'].values.reshape(-1,1)
        # test
        test_df = data.iloc[valtest:]
        test_df['keep'] = test_df.apply(lambda x: (x['UserID'] in train_user_set) and (x['MovieID'] in train_item_set), axis=1)
        test_df = test_df[test_df['keep']==True]
        test_df.drop(columns=['keep'], inplace=True)
        test_df.reset_index(drop=True,inplace=True)
        print(len(train_df), len(val_df), len(test_df))
        test_genres_list = list(map(split, test_df["Genres"].values))
        test_dict = {name: test_df[name] for name in sparse_features}
        test_dict["genres"] = pad_sequences(test_genres_list, maxlen=max_len)
        testx, testy = [test_dict[feature] for feature in feature_index], test_df['Rating'].values.reshape(-1,1)
        
        # return data, linear_feature_columns, dnn_feature_columns, sparse_features, genres_list, target
        train_loader = DataLoader(dataset=train_tensor_data, shuffle=True, batch_size=256, num_workers=0)
        return linear_feature_columns, dnn_feature_columns, train_tensor_data, train_loader, valx, valy, testx, testy, key2index
    elif dataset == 'kuairand':
        parent_dir = '../data/kuairand/'
        video_features = pd.read_csv(parent_dir+'video_features.csv')
        # df_all = pd.read_csv(parent_dir+'interactions.csv')
        df_all = pd.read_csv(parent_dir+'interactions_normal.csv')
        key2index = np.load(parent_dir+'tag_id_dict.npy',allow_pickle=True).item()

        sparse_features = ['user_id','video_id']
        fixlen_feature_columns = [SparseFeat('user_id', 16748), SparseFeat('video_id', 5101)]
        varlen_feature_columns = [VarLenSparseFeat(SparseFeat('tags', vocabulary_size=46+1), maxlen=3)]
        linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
        dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns
        feature_index = build_input_features(linear_feature_columns + dnn_feature_columns)

        # df_all = df_all[df_all['is_rand'] == 0]
        traindf = df_all[df_all['date'] <= 20220422]
        valdf = df_all[df_all['date'] >= 20220423]
        testdf = valdf[valdf['date'] >= 20220501]
        valdf = valdf[valdf['date'] <= 20220430]

        traindf = traindf[['user_id', 'video_id', 'is_click']]
        traindf = pd.merge(traindf, video_features, how='left', on='video_id')
        traindf = shuffle(traindf)
        traindf.reset_index(drop=True, inplace=True)
        traindata = torch.from_numpy(traindf.values)
        trainx, trainy = traindata[:,[0,1,3,4,5]], traindata[:,2].reshape(-1,1)
        train_tensor_data = Data.TensorDataset(trainx, trainy)
        train_loader = DataLoader(dataset=train_tensor_data, shuffle=False, batch_size=256, num_workers=2)

        valdf = valdf[['user_id', 'video_id', 'is_click']]
        valdf = pd.merge(valdf, video_features, how='left', on='video_id')
        valdf = shuffle(valdf)
        valdf.reset_index(drop=True, inplace=True)
        valdata = torch.from_numpy(valdf.values)
        valx, valy = valdata[:,[0,1,3,4,5]], valdata[:,2].reshape(-1,1)

        testdf = testdf[['user_id', 'video_id', 'is_click']]
        testdf = pd.merge(testdf, video_features, how='left', on='video_id')
        testdf = shuffle(testdf)
        testdf.reset_index(drop=True, inplace=True)
        testdata = torch.from_numpy(testdf.values)
        testx, testy = testdata[:,[0,1,3,4,5]], testdata[:,2].reshape(-1,1)
        
        print(len(traindf), len(valdf), len(testdf))
        return linear_feature_columns, dnn_feature_columns, train_tensor_data, train_loader, valx, valy, testx, testy, key2index
    elif dataset == 'book':
        parent_dir = '../data/book/'
        key2index = np.load(parent_dir+'category_id_dict.npy',allow_pickle=True).item()

        sparse_features = ['UserID','ItemID']
        fixlen_feature_columns = [SparseFeat('UserID', 117018), SparseFeat('ItemID', 79199)]
        varlen_feature_columns = [VarLenSparseFeat(SparseFeat('categories', vocabulary_size=486+1), maxlen=5)]
        
        linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
        dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns
        feature_index = build_input_features(linear_feature_columns + dnn_feature_columns)

        # traindata = torch.load(parent_dir+'train.pt')
        # trainx, trainy = traindata[:,[0,1,3,4,5,6,7]], traindata[:,2].reshape(-1,1)
        # train_tensor_data = Data.TensorDataset(trainx, trainy)
        # train_loader = DataLoader(dataset=train_tensor_data, shuffle=False, batch_size=256, num_workers=2)
        train_tensor_data, train_loader = None, None

        # As the train data for book dataset is too big, it's inconvenient to upload it to github, so we set the corresponding train_loder to None.
        # But this operation doesn't affect the reproduction.

        valdata = torch.load(parent_dir+'val.pt')
        valx, valy = valdata[:,[0,1,3,4,5,6,7]], valdata[:,2].reshape(-1,1)
        testdata = torch.load(parent_dir+'test.pt')
        testx, testy = testdata[:,[0,1,3,4,5,6,7]], testdata[:,2].reshape(-1,1)

        return linear_feature_columns, dnn_feature_columns, train_tensor_data, train_loader, valx, valy, testx, testy, key2index
    elif dataset == 'KuaiRand_DT':
        parent_dir = '../data/kuairand/'
        video_features = pd.read_csv(parent_dir+'video_features.csv')
        # df_all = pd.read_csv(parent_dir+'interactions.csv')

        key2index = np.load(parent_dir+'tag_id_dict.npy',allow_pickle=True).item()

        sparse_features = ['user_id','video_id']
        fixlen_feature_columns = [SparseFeat('user_id', 16748), SparseFeat('video_id', 5101)]
        varlen_feature_columns = [VarLenSparseFeat(SparseFeat('tags', vocabulary_size=46+1), maxlen=3)]
        linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
        dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns
        feature_index = build_input_features(linear_feature_columns + dnn_feature_columns)

        # testdf = df_all[df_all['is_rand'] == 1]
        testdf = pd.read_csv(parent_dir+'interactions_random.csv')
        valdf = testdf[testdf['date'] >= 20220423]
        valdf = valdf[valdf['date'] <= 20220430]
        testdf = testdf[testdf['date'] >= 20220501]
        df_all = pd.read_csv(parent_dir+'interactions_normal.csv')
        traindf = df_all[df_all['date'] <= 20220422]
        
        traindf = traindf[['user_id', 'video_id', 'is_click']]
        traindf = pd.merge(traindf, video_features, how='left', on='video_id')
        traindf = shuffle(traindf)
        traindf.reset_index(drop=True, inplace=True)
        traindata = torch.from_numpy(traindf.values)
        trainx, trainy = traindata[:,[0,1,3,4,5]], traindata[:,2].reshape(-1,1)
        train_tensor_data = Data.TensorDataset(trainx, trainy)
        train_loader = DataLoader(dataset=train_tensor_data, shuffle=False, batch_size=256, num_workers=2)

        valdf = valdf[['user_id', 'video_id', 'is_click', 'date']]
        valdf = pd.merge(valdf, video_features, how='left', on='video_id')
        valdf = shuffle(valdf)
        valdf.reset_index(drop=True, inplace=True)
        valdate = valdf['date'].values
        valdf.drop(columns='date',inplace=True)
        valdata = torch.from_numpy(valdf.values)
        valx, valy = valdata[:,[0,1,3,4,5]], valdata[:,2].reshape(-1,1)

        testdf = testdf[['user_id', 'video_id', 'is_click']]
        testdf = pd.merge(testdf, video_features, how='left', on='video_id')
        testdf = shuffle(testdf)
        testdf.reset_index(drop=True, inplace=True)
        testdata = torch.from_numpy(testdf.values)
        testx, testy = testdata[:,[0,1,3,4,5]], testdata[:,2].reshape(-1,1)
        
        print(len(traindf), len(valdf), len(testdf))
        return linear_feature_columns, dnn_feature_columns, train_tensor_data, train_loader, valx, valy, valdate, testx, testy, key2index
    else:
        raise NotImplementedError