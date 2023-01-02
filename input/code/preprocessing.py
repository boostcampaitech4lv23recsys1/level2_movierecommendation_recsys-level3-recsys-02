import pandas as pd
from functools import reduce
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re


def main():
    
    years = pd.read_csv('~/input/data/train/years.tsv', sep = '\t')
    directors = pd.read_csv('~/input/data/train/directors.tsv', sep = '\t')
    genres = pd.read_csv('~/input/data/train/genres.tsv', sep = '\t')
    # titles = pd.read_csv('~/input/data/train/titles.tsv', sep = '\t')
    writers = pd.read_csv('~/input/data/train/writers.tsv', sep = '\t')
    features = [years, directors, genres, writers]
    
    # genre only
    def default() : 
        genres_df = pd.read_csv("../data/train/genres.tsv", sep="\t")
        array, index = pd.factorize(genres_df["genre"])
        genres_df["genre"] = array
        genres_df.groupby("item")["genre"].apply(list).to_json(
            "data/Ml_item2attributes.json"
        )

    def read_tsv(data_path):
        return pd.read_csv(data_path, sep='\t')

    # append to list
    def preprocessing_yhw_1():
        # director, writer 작품 제일 많은 사람 append
        
        print('^-^-'*10)
        print('Data Processing..')
        
        def max_prod(a):
            all_prod = list()
            for i in a:
                all_prod += i
            prod_dict = dict()
            key, cnt = np.unique(all_prod, return_counts= True)
            for i in range(key.shape[0]):
                prod_dict[key[i]] = cnt[i]
            def max_prod_(a):
                max_d = a[0]
                for i in a:
                    if prod_dict[max_d] < prod_dict[i]:
                        max_d = i
                return max_d
            return [max_prod_(i) for i in a]

        def max_dir_apl(df_directors, df_writers, dir_ = True, wrt_ = True, ):
            df_directors = df_directors.groupby(by = ['item'])['director'].apply(list).reset_index(name='director')
            if dir_:        
                df_directors['director'] = max_prod(df_directors['director'])
                df_directors.head()
            
            df_writers = df_writers.groupby(by = ['item'])['writer'].apply(list).reset_index(name='writer')
            if wrt_:
                df_writers['writer'] = max_prod(df_writers['writer'])
                df_writers.head()
            return df_directors, df_writers   

        def list_str(a):
            return [str(a)] 
        
        df_writers = read_tsv('input/data/train/writers.tsv')
        df_directors = read_tsv('input/data/train/directors.tsv')
        df_genres = read_tsv('input/data/train/genres.tsv')
        df_years = read_tsv('input/data/train/years.tsv')
        # df_titles = read_tsv('input/data/train/titles.tsv')
        df_years['year'] = df_years['year'].astype('str')

        df_directors , df_writers = max_dir_apl(df_directors, df_writers, True, True)

        le_prd = LabelEncoder()
        le_prd.fit(np.unique(np.append((np.append(df_writers['writer'], df_directors['director'])), np.append(df_years['year'], df_genres['genre']))))

        df_genres = df_genres.groupby(by = ['item'])['genre'].apply(list).reset_index(name = 'genre')

        df_total = pd.DataFrame({'item' : np.array(range(119146))})

        # df_total = pd.merge(left = df_total, right = df_titles, how = 'left', on = 'item')
        df_total = pd.merge(left = df_total, right = df_directors, how = 'left', on = 'item')
        df_total = pd.merge(left = df_total, right = df_writers, how = 'left', on = 'item')
        df_total = pd.merge(left = df_total, right = df_years, how = 'left', on = 'item')
        df_total = pd.merge(left = df_total, right = df_genres, how = 'left', on = 'item')

        df_total = df_total.dropna()

        df_total['year'] = df_total['year'].apply(list_str)
        df_total['director'] = df_total['director'].apply(list_str)
        df_total['writer'] =df_total['writer'].apply(list_str)
        df_total['genre'] = df_total['genre'].apply(list)

        df_total['total'] = df_total['director'] + df_total['writer'] + df_total['year'] + df_total['genre']

        df_total = df_total.drop(['director', 'writer', 'year', 'genre'], axis = 1)
        df_total = df_total['total'].apply(le_prd.transform)
        
        df_total.to_json(
            "data/Ml_item2attributes_yhw_1.json"
        )
        print('Finished!')
        print('^-^-'*10)
        
        
    def preprocessing_yhw_2():

        # all append
        
        print('^-^-'*10)
        print('Data Processing..')
        
        def list_str(a):
            return [a] 
        df_writers = read_tsv('input/data/train/writers.tsv')
        df_directors = read_tsv('input/data/train/directors.tsv')
        df_genres = read_tsv('input/data/train/genres.tsv')
        df_years = read_tsv('input/data/train/years.tsv')
        # df_titles = read_tsv('input/data/train/titles.tsv')

        df_years['year'] = df_years['year'].astype('str')

        le_prd = LabelEncoder()
        le_prd.fit(np.unique(np.append((np.append(df_writers['writer'], df_directors['director'])), np.append(df_years['year'], df_genres['genre']))))

        df_directors = df_directors.groupby(by = ['item'])['director'].apply(list).reset_index(name = 'director')                             
        df_writers = df_writers.groupby(by = ['item'])['writer'].apply(list).reset_index(name = 'writer')
        df_genres = df_genres.groupby(by = ['item'])['genre'].apply(list).reset_index(name = 'genre')
        df_years['year'] = df_years['year'].apply(list_str)
        
        print('Finished!')
        print('^-^-'*10)

        df_total = pd.DataFrame({'item' : np.array(range(119146))})

        # df_total = pd.merge(left = df_total, right = df_titles, how = 'left', on = 'item')
        df_total = pd.merge(left = df_total, right = df_directors, how = 'left', on = 'item')
        df_total = pd.merge(left = df_total, right = df_writers, how = 'left', on = 'item')
        df_total = pd.merge(left = df_total, right = df_years, how = 'left', on = 'item')
        df_total = pd.merge(left = df_total, right = df_genres, how = 'left', on = 'item')

        df_total = df_total.dropna()

        df_total['total'] = df_total['director'] + df_total['writer'] + df_total['year'] + df_total['genre']
        df_total['total'] = df_total['total'].apply(le_prd.transform)
        df_total = df_total.drop(['director', 'writer', 'year', 'genre'], axis = 1)
        
        df_total.to_json(
            "data/Ml_item2attributes_yhw_2.json"
        )
        print('Finished!')
        print('^-^-'*10)

    def preprocessing_pkj():

        print('^-^-'*10)
        print('Data Processing..')

        ti = pd.read_csv('../data/train/titles.tsv', sep='\t')
        ge = pd.read_csv("../data/train/genres.tsv", sep="\t")
        def year(s):
            if s == 'Fawlty Towers (1975-1979)':
                return 1975
            s = s.split('(')
            answer = []
            for i in s:
                if ')' not in i:
                    continue
                a = re.sub(r'[^0-9]', '', i)
                if len(a) == 4:
                    return int(a)
        ti['year'] = ti['title'].apply(year)
        ti = ti[['item', 'year']]

        item2label = {j:i+1 for i,j in enumerate(sorted(ti['item'].unique()))}
        ti['item'] = ti['item'].map(item2label)
        item2label = {j:i+1+18 for i,j in enumerate((ti['year'].unique()))}
        ti['year'] = ti['year'].map(item2label)

        item2label = {j:i+1 for i,j in enumerate(sorted(ge['item'].unique()))}
        ge['item'] = ge['item'].map(item2label)

        array, index = pd.factorize(ge["genre"])
        ge["genre"] = [i+1 for i in array]

        g = ge.groupby("item")["genre"].apply(list).to_frame()

        g = pd.merge(g, ti, on='item', how='left')

        value = []

        for i in range(len(g)):
            value.append(g.loc[i].genre + [g.loc[i].year])
        g['genre'] = value
        g = g.set_index(g.item)
        g = g['genre']
        g.to_json(
                "../data/train/Ml_item2attributes.json"
            )
        
    # 제외하고 싶은 feature를 선택해서 item2attributes.json 파일 생성
    def preprocessing_mbk(feature=None) : # default = all features
        years = pd.read_csv('~/input/data/train/years.tsv', sep = '\t')
        directors = pd.read_csv('~/input/data/train/directors.tsv', sep = '\t')
        genres = pd.read_csv('~/input/data/train/genres.tsv', sep = '\t')
        # titles = pd.read_csv('~/input/data/train/titles.tsv', sep = '\t')
        writers = pd.read_csv('~/input/data/train/writers.tsv', sep = '\t')
        features = [years, directors, genres, writers]
        
        if feature != None : 
            dfs = list(set(features)-set(feature))
        else : 
            dfs = features
            
        merge = reduce(lambda left,right: pd.merge(left, right, how = 'outer', on = 'item'), dfs)
        
        for col in merge.columns : 
            array, index = pd.factorize(merge[col])
            merge[col] = array
            
        lst=[]
        for feat in ['year','director','genre','writer'] : 
            lst.append(merge.groupby('item')[feat].apply(lambda x:list(set(x))))
            
        merge_attr = reduce(lambda left, right : left+right, lst)
        merge_attr.name = 'attributes'
        merge_attr.columns = ['attributes']
        
        merge = pd.merge(left=merge, right=merge_attr, on='item', how='inner')
        
        return merge.groupby('item')['attributes'].apply(lambda x:list(x)[0]).to_json(
            "data/ML_item2attributes_all.json"
        )

    # preprocessing_ohj.함수명으로 함수 호출 가능
    class preprocessing_ohj:
        # timestamp -> year, month, daty, hour, minute, second로 나누기
        def split_time(train_df):
            import time
            datetime_df = train_df['time'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))
            datetime_df = datetime_df.str.split(" ", expand=True)

            datetime_df.columns = ['date', 'time']

            date_df = datetime_df['date'].str.split("-", expand=True)
            date_df.columns = ['year', 'month', 'day'] 

            time_df = datetime_df['time'].str.split(":", expand=True)
            time_df.columns = ['hour', 'minute', 'second'] 

            datetime_df = pd.concat([datetime_df, date_df], axis=1)
            datetime_df = pd.concat([datetime_df, time_df], axis=1)

            print(list(datetime_df))
            train_df = pd.concat([train_df, datetime_df], axis=1)

            return train_df

        # 모든 아이템 feature 합치기 
        def create_item_df(data_path):
            import os

            year_data = pd.read_csv(os.path.join(data_path, 'years.tsv'), sep='\t')
            writer_data = pd.read_csv(os.path.join(data_path, 'writers.tsv'), sep='\t')
            title_data = pd.read_csv(os.path.join(data_path, 'titles.tsv'), sep='\t')
            genre_data = pd.read_csv(os.path.join(data_path, 'genres.tsv'), sep='\t')
            director_data = pd.read_csv(os.path.join(data_path, 'directors.tsv'), sep='\t')

            item_df = pd.merge(year_data, writer_data, on='item', how='outer')
            item_df = pd.merge(item_df, title_data, on='item', how='outer')
            item_df = pd.merge(item_df, genre_data, on='item', how='outer')
            item_df = pd.merge(item_df, director_data, on='item', how='outer')

            print(list(item_df))

            return item_df

        # data type 수정하기
        def refactor_data_type(year_data, writer_data, director_data):
            year_data['year'] = year_data['year'].astype(int)
            writer_data['writer'] = writer_data['writer'].str[2:]
            director_data['director'] = director_data['director'].str[2:]
            
            return year_data, writer_data, director_data

        # title에 (연도) 제거, ',The' 제거
        def simplify_title(title_data):
            import re

            # 괄호와 괄호 내 문자열 제거
            title_data['title'] = title_data['title'].str.replace(pat = r'\(.*\)|\s-\s.*', repl=r'', regex=True)
            title_data['title'] = title_data['title'].str.replace(pat = r'\, The|\s-\s.*', repl=r'', regex=True)
            title_data['title'] = title_data['title'].str.strip()

            return title_data

    
    
if __name__ == "__main__":
    main()
