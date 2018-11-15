# 475k für train, 25k for test

import pandas as pd
from itertools import product, chain, combinations
import random
random.seed(1337, 2)


SAME_SPEAKER = 1
DIFFERENT_SPEAKER = 0

from vox_utils import get_dataset, TRAIN, DEV, TEST


def write_same_pairs_file(split_frame, train_set_size, num_per_poi, csv_name):
    grouped = split_frame.groupby('id')
    pairs_list = list()
    i = 0
    for name, group in grouped:
        j = 0
        for index in list(combinations(group.index, 2)):
            pairs_list.append(
                [SAME_SPEAKER, group.loc[index[0], 'path'], group.loc[index[1], 'path']])

            i += 1
            j += 1
            if j >= num_per_poi:
                break
        if i >= train_set_size / 2:
            break
    same_speaker_df = pd.DataFrame(data=pairs_list, columns=[
                                   'y_label', 'utterance_a', 'utterance_b'])
    same_speaker_df.to_csv(csv_name)

def create_combis(use_test_split=False):
    dataset = pd.DataFrame(columns=['y_label', 'utterance_a', 'utterance_b'])
    df = get_dataset()
    if use_test_split:
        df = df[df['split'] == TEST]
    else:   
        df = df[df['split'] != TEST]

    criteria = set(df[['Nationality', 'Gender']].itertuples(index=False))

    for criterium in criteria:
        similars = df[(df['Nationality'] == criterium.Nationality) & (df['Gender'] == criterium.Gender)]
        combis = (x for x in combinations(similars.itertuples(), 2) if x[0].speaker_id != x[1].speaker_id)
        for combi in combis:
            dataset.append({
                'y_label': DIFFERENT_SPEAKER,
                'utterance_a': combi[0].path,
                'utterance_b': combi[1].path
            }, ignore_index=True)

    return dataset

if __name__ == "__main__":
    df = create_combis()
    df.to_csv('similar_train.csv')
    df = create_combis(True)
    df.to_csv('similar_test.csv')
