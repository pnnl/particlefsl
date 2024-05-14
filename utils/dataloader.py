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
import tensorflow as tf
from utils import data

def make_dataset(args, path_to_single_dataset):
    """Prepares training and validation data generators for a single database.

    Args:
      args: Command-line arguments.
      path_to_single_dataset: Path to .hdf5 dataset.

    Returns:
      Training and validation data generators.
    """
    # make keys
    data.gen_key_pairs(args, path_to_single_dataset)
    
    # making training set
    dg = data.PairDataset(args, 
                          keyset=op.join(args.savedir, args.model_name, 
                                         path_to_single_dataset.split('/')[-1].replace('.hdf5','_train')),
                          infer_database=path_to_single_dataset, 
                          shuffle=True)

    # generate validation set
    dgv = data.PairDataset(args, 
                          keyset=op.join(args.savedir, args.model_name, 
                                         path_to_single_dataset.split('/')[-1].replace('.hdf5','_val')),
                          infer_database=path_to_single_dataset, 
                          shuffle=False)

    # create data generators 
    if args.sem:
        return [tf.data.Dataset.from_generator(dg, 
                                              output_signature=(tf.TensorSpec(shape=(4,128,128), dtype=tf.float32), 
                                              tf.TensorSpec(shape=(), dtype=tf.int16))),
                tf.data.Dataset.from_generator(dgv, 
                                               output_signature=(tf.TensorSpec(shape=(4,128,128), dtype=tf.float32), 
                                               tf.TensorSpec(shape=(), dtype=tf.int16)))]
    else:
        
        return [tf.data.Dataset.from_generator(dg,
                                              output_signature=(tf.TensorSpec(shape=(2,args.cutoff-args.cutoff_start,1), 
                                                                              dtype=tf.float32), 
                                              tf.TensorSpec(shape=(), dtype=tf.int16))), 
                tf.data.Dataset.from_generator(dgv, 
                                               output_signature=(tf.TensorSpec(shape=(2,args.cutoff-args.cutoff_start,1), 
                                                                               dtype=tf.float32), 
                                               tf.TensorSpec(shape=(), dtype=tf.int16)))]

def func(x):
    return x


def get_sets(args):
    """Prepares data input pipelines, combining multiple databases.

    Args:
      args: Command-line arguments.

    Returns:
      Training and validation sets, ready for training.
    """
    # apply auto sharding
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    # interleave datasets
    N_DATASETS_TO_INTERLEAVE=len(args.database)       
    alldatasets = [make_dataset(args, args.database[i]) for i in range(N_DATASETS_TO_INTERLEAVE)]
    traindatasets = [sets[0] for sets in alldatasets]

    # prepare train set
    train_set = tf.data.Dataset.from_tensor_slices(traindatasets)
    train_set = train_set.interleave(tf.autograph.experimental.do_not_convert(func), 
                                     cycle_length=N_DATASETS_TO_INTERLEAVE, num_parallel_calls=tf.data.AUTOTUNE)
    train_set = train_set.with_options(options)
    train_set = train_set.batch(args.batch_size)
    train_set = train_set.prefetch(tf.data.AUTOTUNE)

    # prepare validation set
    valdatasets = [sets[1] for sets in alldatasets]
    val_set = tf.data.Dataset.from_tensor_slices(valdatasets)
    val_set = val_set.interleave(tf.autograph.experimental.do_not_convert(func),
                                 cycle_length=N_DATASETS_TO_INTERLEAVE, num_parallel_calls=tf.data.AUTOTUNE)
    val_set = val_set.with_options(options)
    val_set = val_set.batch(args.batch_size)
    val_set = val_set.prefetch(tf.data.AUTOTUNE)
    
    return train_set, val_set