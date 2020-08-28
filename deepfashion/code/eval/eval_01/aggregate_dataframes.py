#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import glob
from multiprocessing import Pool
from contextlib import closing
import tqdm


def process_file(fname):
    df = pd.read_csv(fname, index=False, header=True)
    return df


def main():
    f_list = glob.glob("*/part_ious.csv")
    dfs = []
    with closing(Pool(8)) as p:
        for df in tqdm.tqdm(p.imap(process_file, f_list)):
            dfs.append(df)

    final_df = pd.concat(dfs)
    final_df.to_csv("ious.csv", index=False, header=True)


if __name__ == "__main__":
    main()
