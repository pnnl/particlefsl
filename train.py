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

import os
import os.path as op
import logging

import json

import tensorflow as tf

#### LOAD UTILS ####
from utils import dataloader, modelloader


import argparse

parser = argparse.ArgumentParser()
# directory parameters
parser.add_argument('--savedir', required=True, type=str, help='path to dir to save training')
parser.add_argument('--model_name', required=True, type=str, help='name for model dir')
# training parameters
parser.add_argument('--mode', default='train', choices=['train', 'eval'], help='train or eval')
parser.add_argument('--parallel', dest='parallel', action='store_true', help='flag for multigpu training')
parser.add_argument('--batch_size', default=128, type=int, help='training batch size')
parser.add_argument('--learning-rate-start', default=1e-3, type=float, help='Learning rate start')
parser.add_argument('--learning-rate-end', default=1e-4, type=float, help='Learning rate end')
parser.add_argument('--epochs', default=1000, type=int, help='training epochs')
parser.add_argument('--grad-clip', default=1.0, type=float, help='gradient clipping')
# model parameters
parser.add_argument('--dropout', default=0.3, type=float, help='dropout (must be >0 for UQ)')
parser.add_argument('--activation', default='relu', type=str, help='activation')
parser.add_argument('--prepool', dest='prepool', action='store_true', help='apply prepooling')
parser.add_argument('--prepool_size', default=3, type=int, help='prepooling size')
parser.add_argument('--pool_size', default=2, type=int, help='pooling size')
parser.add_argument('--entropy', default='square', choices=['square', 'abs'], help='comparison method for entropy network')
parser.add_argument('--basemodel', default='None', type=str, help='Path to model to restart training from')
# data parameters
parser.add_argument('--database', required=True, nargs='+',type=str, help='path to hdf5 database')
parser.add_argument('--validation_split', default=0.1, type=str, help='fraction of data to use for validation split')
parser.add_argument('--simelim', dest='simelim', action='store_true', help='use simelim to refine training set')
parser.add_argument('--simelim_percentile', default=2.2, type=float, help='simelim percentile to use (between 0 and 100)')
parser.add_argument('--cutoff', default=2500, type=int, help='eds spectra cutoff')
parser.add_argument('--cutoff_start', default=0, type=int, help='front cutoff for eds spectra')
parser.add_argument('--subsample', default=-1, type=int, help='downsample training set to this value')
parser.add_argument('--max_class_sample', default=500, type=int, help='downsample each class to this value')
parser.add_argument('--sem', dest='sem', action='store_true', help='use SEM images for training')
parser.add_argument('--pair_ratio', default=1.0, type=float, help='ratio of different class pairs in 1:X (same:different) format')
args = parser.parse_args()


# set up directory structure
if not op.isdir(op.join(args.savedir, args.model_name)):
    os.mkdir(op.join(args.savedir, args.model_name))
    os.mkdir(op.join(args.savedir, args.model_name, 'inference'))

# save arguments
with open(op.join(args.savedir, args.model_name, 'args.json'), 'wt') as f:
    json.dump(vars(args), f, indent=4)

# set up logger
logging.basicConfig(filename=op.join(args.savedir, args.model_name,'training.log'), level=logging.INFO)


# create data loaders
train_set, val_set = dataloader.get_sets(args)

# create model
model, callbacks = modelloader.get_model(args)

# train model
history = model.fit(train_set,
                    epochs = args.epochs,
                    verbose = 0,
                    validation_data = val_set,
                    callbacks = callbacks
                    )



