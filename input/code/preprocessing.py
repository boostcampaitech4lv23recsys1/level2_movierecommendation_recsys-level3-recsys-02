import pandas as pd
from functools import reduce


def main():
    # genre
    genres_df = pd.read_csv("../data/train/genres.tsv", sep="\t")
    array, index = pd.factorize(genres_df["genre"])
    genres_df["genre"] = array
    genres_df.groupby("item")["genre"].apply(list).to_json(
        "data/Ml_item2attributes.json"
    )
    
    years = pd.read_csv('~/input/data/train/years.tsv', sep = '\t')
    directors = pd.read_csv('~/input/data/train/directors.tsv', sep = '\t')
    genres = pd.read_csv('~/input/data/train/genres.tsv', sep = '\t')
    # titles = pd.read_csv('~/input/data/train/titles.tsv', sep = '\t')
    writers = pd.read_csv('~/input/data/train/writers.tsv', sep = '\t')
    
    # 제외하고 싶은 feature를 선택해서 item2attributes.json 파일 생성
    def select_features(feature=None) : # default = all features
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
        
    select_features()


if __name__ == "__main__":
    main()
