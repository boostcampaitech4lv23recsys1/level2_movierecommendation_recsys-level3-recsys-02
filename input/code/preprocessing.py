import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def main():
    default()

def read_tsv(data_path):
    return pd.read_csv(data_path, sep='\t')

# main
def default():
    genres_df = pd.read_csv("../data/train/genres.tsv", sep="\t")
    array, index = pd.factorize(genres_df["genre"])
    genres_df["genre"] = array
    genres_df.groupby("item")["genre"].apply(list).to_json(
        "data/Ml_item2attributes.json"
    )
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
    
    
if __name__ == "__main__":
    main()
