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

from PIL import Image
import pandas as pd
import numpy as np
import itertools
import os
import os.path as op
import logging
import random
import h5py
from sklearn.model_selection import train_test_split
import csv
import tensorflow as tf
import math

def norm_data(image_path, size=2500, start=0):
    """Extract EDS spectrum from tif, reshape, and normalize (max=1).

    Args:
      image_path (str): Path to tif containing data.
      size (int): Desired length of spectrum.
      start (int): Index before which spectrum is trimmed.

    Returns:
      Trimmed, normalized spectrum with shape (size, 1)
    """
    
    # extract spectrum
    with Image.open(image_path) as img:
        spectra = np.array(img.tag[33618])    
        
    # trim
    spectra = spectra[start:start+size]
    
    # normalize by dividing by max, reshape to (size, 1)
    return np.expand_dims(spectra/spectra.max(), axis=1)

def read_raw_data(file):
    """Read csv of raw data.

    Args:
      file: Path to hdz of raw data

    Returns:
      Dataframe with sample information.
    """

    # read hdz file to get column headers
    with open(file+'.hdz') as f:
        lines = f.readlines()
    lines=[l.replace('\n','').replace('\t',' ') for l in lines]

    # filter to start of header
    start_idx=[i for i,l in enumerate(lines) if l=='PART# 1 INT16'][0]
    header=[l.split(' ')[0] for l in lines[start_idx:]]

    # read pxz column to get data
    df = pd.read_csv(file+'.pxz', delimiter='\t', names=header)

    # replace PSEM_CLASS with name of class
    class_map={int(l.replace('CLASS','').split('=')[0]):l.replace('CLASS','').split('=')[1] for l in lines
               if l[:5]=='CLASS' and l[:7]!='CLASSES'}
    df['PSEM_CLASS']=df['PSEM_CLASS'].apply(lambda x: class_map[x])
    
    return df

def gen_database(raw_data_dir, class_file, database):
    """Creates hdf5 of data and csv of particle info for a sample.
    
    Args:
       raw_data_dir: Top-level of raw data dir
       class_file: File name without file extension (hdz/pxz pair)
       database: Path to store database without file extension

    """
    # open the file in append mode
    hf = h5py.File(database+'.hdf5', 'a')

    present_samples=[]
    for j in os.listdir(op.join(raw_data_dir, 'MAG0')):
        if not j.endswith('.tif'):
            continue

        sample_name = str(int(j.replace('.tif','')))
        present_samples.append(sample_name)

        track = os.path.join(raw_data_dir, 'MAG0', j)
        subgrp = hf.create_group(sample_name)

        ### Save EDS spectrum
        spectra = norm_data(track)
        dset = subgrp.create_dataset('EDS', data=spectra)

        ### Save SEM images
        tiffstack= Image.open(track)
        tiffstack.load()

        for i in range(tiffstack.n_frames):
            tiffstack.seek(i)
            dset = subgrp.create_dataset('SEM'+str(i), data=np.asarray(tiffstack))

        tiffstack.close()

    hf.close()

    #filter csv to exclude missing images
    present_samples = [int(ps) for ps in present_samples]
    sample = read_raw_data(op.join(raw_data_dir, class_file))
    sample = sample.loc[sample['PART#'].isin(present_samples)]
    sample.to_csv(database+'.csv', index=False)


####GEN KEYPAIRS####
def gen_key_pairs(args, database):
    """Create pairwise keys for training, validation, and test sets.

    Args:
      args: Command-line arguments.
      database: Path to hdf5 database
      
    """
        
    # check if training and validation set keys have been generated
    if op.isfile(op.join(args.savedir, args.model_name, 'train_keys.csv')) and op.isfile(op.join(args.savedir, args.model_name, 'val_keys.csv')):
        logging.info("...loading premade keys")
        return
    else:
        logging.info("...generating new keys")
        # generate keys from random sampling
        train_keys, val_keys, test_keys = pair_particles(args, database)

    # save keys to model dir
    train_keys.to_csv(op.join(args.savedir, args.model_name, database.split('/')[-1].replace('.hdf5','_train_keys.csv')), index=False)
    val_keys.to_csv(op.join(args.savedir, args.model_name, database.split('/')[-1].replace('.hdf5','_val_keys.csv')), index=False)
    test_keys.to_csv(op.join(args.savedir, args.model_name, database.split('/')[-1].replace('.hdf5','_test_keys.csv')), index=False)
    return


### SIMELIM KEY GEN ###
def match_different(row, df):
    """Sample a particle of a different PSEM class than that of row.

    Args:
      row: Row in dataframe of particle to be paired.
      database: Dataframe of particles for pairing.
      
    Returns:
      Particle ID and PSEM class
    """    
    
    tmp = df.loc[df.PSEM_CLASS!=row.class_l]
    
    # randomly sample one row
    pair = tmp.sample(n=1).iloc[0]
    
    # resample if PSEM classes the same
    count=0
    while pair.PSEM_CLASS == row.class_l:
        # if partner isn't found after 50 tries, return None
        if count > 50:
            return -1, 'None'
        pair = tmp.sample(n=1).iloc[0]
        count+=1
    
    return pair['PART#'], pair['PSEM_CLASS']

def pair_particles(args, path_to_single_dataset):
    """Pair particles and create splits.

    Args:
      args: Command-line arguments.
      path_to_single_dataset: Path to .hdf5 dataset.
      
    Returns:
      Pair dataframes for training, validaiton, and test splits.
    """      
    # load df
    df = pd.read_csv(path_to_single_dataset.replace('.hdf5','.csv'))
    
    if args.simelim:
        simelim_dir=op.join('/'.join(path_to_single_dataset.split('/')[:-1]),'simelim')
        # collect simelim scores
        score_files = [x for x in os.listdir(simelim_dir) if x.endswith('.csv')]
        logging.info(f"{len(score_files)} simelim files found in {simelim_dir}")

        sf = pd.DataFrame()

        # filter pairs by percentile-based simelim score cutoff
        for fil in score_files:
            tmp = pd.read_csv(op.join(simelim_dir, fil))

            if len(tmp)>=1:
                cutoff_score=np.percentile(tmp.score, args.simelim_percentile)
                tmp = tmp[tmp.score.ge(cutoff_score)].copy()
            if args.max_class_sample < len(tmp):
                tmp = tmp.sample(n=args.max_class_sample, replace=False, axis=0)

            sf = pd.concat([sf,tmp], ignore_index=True, sort=False)

        logging.info("...simelim percentile cutoff applied")

        # collect same pairs
        sf=sf[['x_l','x_r','class_l','class_r']]
        sf['y']=1
    else:
        # get all classes in sample
        all_classes = list(set(self.sample['PSEM_CLASS']))

        # generate pairs
        same_combos = list(itertools.chain.from_iterable([list(itertools.combinations(self.sample.loc[self.sample.PSEM_CLASS==c]['PART#'].tolist(),2)) for c in all_classes]))

        sf=pd.DataFrame.from_records(same_combos, columns=['x_l', 'x_r'])
        sf['y']=1

    logging.info(f"...same class pairs generated: {len(sf)} total")

    # collect different pairs
    # randomly sample x_l from dataframe
    dif = df.sample(n=int(len(sf)*args.pair_ratio), 
                    replace=True, axis=0)[['PART#','PSEM_CLASS']]
    df.reset_index(inplace=True, drop=True)
    dif['x_l']=dif['PART#']
    dif['class_l']=dif['PSEM_CLASS']
    dif.drop(['PART#','PSEM_CLASS'],inplace=True,axis=1)
    
    # for each row, randomly sample a particle of a different PSEM class
    dif[['x_r','class_r']]=dif.apply(lambda x: match_different(x, df), axis=1, result_type='expand')
    
    # remove rows where pairs could not be found
    dif = dif.loc[dif['x_r']>-1].copy()
    dif['y']=0

    logging.info(f"...different class pairs generated: {len(dif)} total")

    # check
    if len(dif.loc[dif.class_l==dif.class_r])!=0:
        logging.info('problem with processing -- different pairs are same class')

    pairs = pd.concat([sf,dif], ignore_index=True, sort=False)
    
    # split pairs into train, val, test split even val and test
    frac = 1 - (2*args.validation_split)
    train=pairs.groupby('y', group_keys=False).apply(lambda x: x.sample(frac=frac))
    test=pairs.drop(train.index)
    val=test.groupby('y', group_keys=False).apply(lambda x: x.sample(frac=0.5))
    test=test.drop(val.index)
        
    return train, val, test


####DATA LOADER####
class PairDataset:
    def __init__(self, args, keyset, shuffle=True, infer_database='None'):
        """Pairwise dataset class for training and inference.

        Args:
          args: Command-line arguments.
          keyset: Dataset split to generate ('train' or 'val').

        """
        self.cutoff_start = args.cutoff_start
        self.cutoff = args.cutoff
        self.sem = args.sem
        self.shuffle = shuffle

        # load keys
        if infer_database!='None':
            self.datafile = h5py.File(infer_database, 'r')
            keys = pd.read_csv(f'{keyset}_keys.csv')
        else:
            self.datafile = h5py.File(args.database, 'r')
            keys = pd.read_csv(op.join(args.savedir, args.model_name, f'{keyset}_keys.csv'))
        
            keys.loc[keys['class_l']==keys['class_r']]['class_l'].value_counts().to_csv(op.join(args.savedir, 
                                                                                                args.model_name,
                                                                                                f'{keyset}_keys_same_pair_counts.csv'))

        keys = keys[['x_l', 'x_r', 'y']].copy()
        
        # reduce size of dataset
        if args.subsample > 0:
            reduce_to = args.subsample if 'train' in keyset else int(args.subsample*args.validation_split)
            if reduce_to < len(keys):
                keys, _ = train_test_split(keys, train_size=reduce_to, random_state=42, stratify=keys[['y']])
                logging.info(f'total {keyset} set after reduce_to = {len(keys)}')
        logging.info(f'total {keyset} set = {len(keys)}')

        self.keys = np.array(list(keys.itertuples(index=False, name=None)))

    def __len__(self):
        return len(self.keys) 

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]

        # collect spectrum data
        spectra_l = self.datafile[str(key[0])]['EDS'][self.cutoff_start:self.cutoff]
        spectra_r = self.datafile[str(key[1])]['EDS'][self.cutoff_start:self.cutoff]

        if self.sem:
            # collect image data
            images_l = self.datafile[str(key[0])]['SEM0']
            images_r = self.datafile[str(key[1])]['SEM0']
            length = images_l.shape[0]*images_l.shape[1]
            
            # create mask for spectra
            mask = np.reshape(np.concatenate([np.zeros(spectra_l.shape[0], dtype=bool), 
                                              np.ones(length-spectra_l.shape[0], dtype=bool)]), 
                              (images_l.shape[0], images_l.shape[1]))
            
            # create masked arrays
            spectra_l = np.ma.array(np.resize(spectra_l, (images_l.shape[0], images_l.shape[1])), mask=mask)
            spectra_r = np.ma.array(np.resize(spectra_r, (images_l.shape[0], images_l.shape[1])), mask=mask)
            return np.stack([spectra_l, spectra_r, images_l, images_r], axis=0), key[2].astype(np.int16)

        else:
            return np.stack([spectra_l, spectra_r], axis=0), key[2].astype(np.int16)


    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

            if i == self.__len__()-1:
                self.on_epoch_end()

