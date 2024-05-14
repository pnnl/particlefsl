"""
This material was prepared as an account of work sponsored by an agency of the
United States Government.  Neither the United States Government nor the United
States Department of Energy, nor Battelle, nor any of their employees, nor any
jurisdiction or organization that has cooperated in the development of these
materials, makes any warranty, express or implied, or assumes any legal
liability or responsibility for the accuracy, completeness, or usefulness or
any information, apparatus, product, software, or process disclosed, or
represents that its use would not infringe privately owned rights.

Reference herein to any specific commercial product, process, or service by
trade name, trademark, manufacturer, or otherwise does not necessarily
constitute or imply its endorsement, recommendation, or favoring by the United
States Government or any agency thereof, or Battelle Memorial Institute. The
views and opinions of authors expressed herein do not necessarily state or
reflect those of the United States Government or any agency thereof.

                 PACIFIC NORTHWEST NATIONAL LABORATORY
                              operated by
                                BATTELLE
                                for the
                   UNITED STATES DEPARTMENT OF ENERGY
                    under Contract DE-AC05-76RL01830
"""

import itertools
from datetime import datetime
import os
import os.path as op
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', required=True, type=str, help='Path to trained model')
parser.add_argument('--database', required=True, type=str, help='hdf5 database corresponding to pair data')
parser.add_argument('--tag', required=True, type=str, help='Unique ID for subset (int)')
parser.add_argument('--n-particles', default=-1, type=int, help='Number of particles to analyze')
parser.add_argument('--n-samples', default=10, type=int, help='Number of samples for MC dropout')
parser.add_argument('--batch-size', default=1024, type=int, help='Batch size for inference')
args = parser.parse_args()

from utils import analysis, modelloader


sample = args.database.split('/')[-1].replace('.hdf5','')
savedir = op.join(args.modeldir, 'inference', sample)

if not op.isdir(savedir):
    os.makedirs(savedir, exist_ok=True)

infer_name = sample + '_' + args.tag
print(f"results saving to {savedir} as {infer_name}_*.csv")

# load trained model
model = modelloader.load_FewShotNetwork(op.join(args.modeldir,'best_model.h5'))
print(f'best_model loaded from {args.modeldir}')

# load data and filter
df=pd.read_csv(args.database.replace('.hdf5','.csv'))
print(f"data loaded from {args.database.replace('.hdf5','.csv')}")

# create random subsample 
if args.n_particles==-1 or args.n_particles>len(df):
    print(f"analyzing FULL dataset (all pairs)")
    tmp = df
else:
    tmp = df.sample(args.n_particles)

nodes = tmp['PART#'].tolist()

keys = analysis.pair_all_particles(tmp)
analysis.sort_pairs(keys)

# load precomputed pairs
precomp_path = op.join(savedir, 'compiled.csv')
if op.isfile(precomp_path):
    comp_flag=True
    precomp = pd.read_csv(precomp_path)
    print(f"loading precomputed pairs from {precomp_path}")

    # check number of MC runs in compiled results
    n_precomp_mc = len([x for x in precomp.columns if 'mc_' in x])
    if n_precomp_mc != args.n_samples:
        print(f'WARNING:: precomputed file had {n_precomp_mc} MC runs -- reverting to this value')
        args.n_samples = n_precomp_mc

    precomp['pair']=precomp.apply(lambda x: tuple(sorted([x.x_l, x.x_r])), axis=1)
    nodes = tmp['PART#'].tolist()
    combos=list(itertools.combinations(nodes,2))
    missing_pairs = list(set(combos) - set(precomp['pair'].tolist()))
    keys = analysis.get_groups(missing_pairs,df)
    analysis.sort_pairs(keys)

    # setup df for prior pair inference
    comp_df=precomp.loc[(precomp['x_low'].isin(keys['x_low']))&(precomp['x_high'].isin(keys['x_high']))]
    print(f"{len(combos)-len(keys)} pairs were precomputed")
    print(f"{len(keys)} left to compute")
else:
    keys = analysis.pair_all_particles(tmp)
    analysis.sort_pairs(keys)
    comp_flag=False
    print("no previous pair inference")

# save keys to be computed
keys.to_csv(op.join(savedir, f'{infer_name}_keys.csv'), index=False)
results_df_all = keys

# memory clean up
del df

# make data loader
infer_set = analysis.make_infer_dataset(args, keyset=op.join(savedir, infer_name))

# run inference n_samples number of times
results_df = pd.DataFrame()
save_path = op.join(savedir,f'{infer_name}_edgelist.csv')
now = datetime.now()
for n in tqdm(range(args.n_samples)):
    y_pred = model.predict(infer_set)

    # save in edgelist format
    tmp = results_df_all[['x_l','x_r','class_l','class_r']].copy()
    tmp['similarity'] = y_pred
    if n == 0:
        tmp.to_csv(save_path, mode='w', index=False, header=True)
    else:
        tmp.to_csv(save_path, mode='a', index=False, header=False)

    # save as columns
    results_df_all[f'mc_{n}'] = y_pred
    print(f'{n}: {(datetime.now()-now).seconds/60:0.1f} mins')

results_df_all.to_csv(op.join(savedir, f'{infer_name}.csv'), index=False)
print(f"{args.n_samples} repeat inferences saved to {savedir}/{infer_name}.csv")

if comp_flag:
    # re-add precomputed pairs to edgelist
    for c in [x for x in comp_df.columns if 'mc_' in x]:
        tmp = comp_df[['x_l','x_r','class_l','class_r',c]]
        tmp.to_csv(save_path, mode='a', index=False, header=False)
    
    # re-add precomputed pairs to results_df_all
    comp_df.to_csv(op.join(savedir, f'{infer_name}.csv'), mode='a', index=False, header=False)

    # add newly computed pair results to precomp
    results_df_all.to_csv(precomp_path, mode='a', index=False, header=False)
else:
    results_df_all.to_csv(precomp_path, index=False)


# stats for repeated MC pairs
results_df = pd.read_csv(save_path)

# create pivot chart by averaging repeated MC pairs
# get similarity means and std dev from repeated MC runs
d1=results_df.groupby(['x_l','x_r'])['similarity'].mean().reset_index()
d2=results_df.groupby(['x_l','x_r'])['similarity'].std().reset_index()
d2['std']=d2['similarity']
d2.drop(['similarity'], axis=1, inplace=True)

# join dataframes
pivot = pd.merge(d1, d2,  on=["x_l", "x_r"])

# merge in class names
pivot = pd.merge(pivot, results_df[['x_l','x_r','class_l','class_r']], on=['x_l','x_r'], how='left').drop_duplicates()

# resave pivot chart
pivot.to_csv(op.join(savedir, f'{infer_name}_edgelist_mean.csv'), index=False)