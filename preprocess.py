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
import glob
from utils import data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, default='./data/raw', help="Top level derectory of raw data repo")
parser.add_argument('--savedir', type=str, default='./data/processed', help="Top level directory to save processed data")
parser.add_argument('--sample', type=str, required=True, help="Sample to process.")
args = parser.parse_args()

print(f"{len(glob.glob(op.join(args.datadir, args.sample, 'MAG0','*.tif')))} particles")

# create savedir
savedir=op.join(args.savedir, args.sample.replace(" ","_"))

# set up directory
if not op.isdir(savedir):
    os.makedirs(savedir, exist_ok=True)

# generate database
data.gen_database(raw_data_dir=op.join(args.datadir, args.sample),
                  class_file='neQuantC6s JORS[GSR+Base]',
                  database=op.join(savedir, args.sample.replace(" ","_")))
