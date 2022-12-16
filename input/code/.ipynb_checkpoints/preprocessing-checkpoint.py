import pandas as pd
import re

def main():
    # genres_df = pd.read_csv("../data/train/genres.tsv", sep="\t")
    # item2label = {j:i+1 for i,j in enumerate(sorted(genres_df['item'].unique()))}
    # genres_df['item'] = genres_df['item'].map(item2label)
    # array, index = pd.factorize(genres_df["genre"])
    # genres_df["genre"] = array
    # genres_df.groupby("item")["genre"].apply(list).to_json(
    #     "../data/train/Ml_item2attributes.json"
    # )
    ti = pd.read_csv('../data/train/titles.tsv', sep='\t')
    ge = pd.read_csv("../data/train/genres.tsv", sep="\t")
    def year(s):
        s = s.split('(')
        answer = []
        for i in s:
            if ')' not in i:
                continue
            a = re.sub(r'[^0-9]', '', i)
            if len(a) == 4:
                return a
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

if __name__ == "__main__":
    main()
