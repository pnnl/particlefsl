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

import os.path as op
import itertools
import json
import pandas as pd
import tensorflow as tf
from utils import data


class train_args:
    """Read train_args from json"""
    def __init__(self, save_dir):
        dictionary = json.load(open(op.join(save_dir,'args.json')))
        for k, v in dictionary.items():
            setattr(self, k, v)
            
### FUNCTIONS TO MAKE PAIRS ###
def get_groups(pairs, sample):
    # get class names
    keys=pd.DataFrame.from_records(pairs, columns=['x_l', 'x_r'])

    keys=keys.merge(sample[['PART#','PSEM_CLASS']], left_on='x_l', right_on='PART#')
    keys['class_l']=keys['PSEM_CLASS']
    keys=keys.drop(['PART#','PSEM_CLASS'], axis=1)
    keys=keys.merge(sample[['PART#','PSEM_CLASS']], left_on='x_r', right_on='PART#')
    keys['class_r']=keys['PSEM_CLASS']
    keys=keys.drop(['PART#','PSEM_CLASS'], axis=1)
    keys['y']=0
    keys.loc[(keys.class_l==keys.class_r), 'y']=1
    return keys

def pair_all_particles(sample):
    # pair all partners
    partn_pairs=list(itertools.combinations(sample['PART#'].tolist(),2))

    # merge into dataframe
    keys = get_groups(partn_pairs,sample)
    
    return keys

def sort_pairs(df):
    # add sorted pairs as columns to key dfs
    df['pair'] = df.apply(lambda x: sorted((x['x_l'],x['x_r'])), axis=1)
    df['x_low'] = df['pair'].apply(lambda x: x[0])
    df['x_high'] = df['pair'].apply(lambda x: x[1])
    df.drop(['pair'], axis=1, inplace=True)           
    
def make_infer_dataset(args, keyset):
    """Prepares inference data generator.

    Args:
      args: Command-line arguments.
      keyset: Name for inference set.

    Returns:
      Inference data generator.
    """
    # load training arguments from model
    model_train_args = train_args(args.modeldir)
    model_train_args.subsample = 0 
    
    # apply auto sharding
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

    # make data loader
    dg = data.PairDataset(model_train_args, keyset=keyset, shuffle=False, infer_database=args.database)
    if model_train_args.sem:
        infer_set = tf.data.Dataset.from_generator(dg, output_signature=(tf.TensorSpec(shape=(4,128,128), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.int16)))
    else:
        infer_set = tf.data.Dataset.from_generator(dg, output_signature=(tf.TensorSpec(shape=(2,model_train_args.cutoff-model_train_args.cutoff_start,1), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.int16)))
    infer_set = infer_set.with_options(options)
    infer_set = infer_set.batch(args.batch_size)
    infer_set = infer_set.prefetch(tf.data.AUTOTUNE)    
    return infer_set
            





