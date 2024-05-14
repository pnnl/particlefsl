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

import sys
import os
import os.path as op
import numpy as np
import pandas as pd
from datetime import datetime
import matchms.filtering as msfilters
from matchms import Spectrum
from spec2vec import SpectrumDocument
from spec2vec.model_building import train_new_word2vec_model
import gensim
from matchms import calculate_scores
from spec2vec import Spec2Vec
import h5py
from multiprocessing import Pool
import itertools
import random
import argparse
from utils import simelim

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, default='./data/processed', help="Top level derectory of processed data directory")
parser.add_argument('--sample', type=str, required=True, help="Name of sample to analyze")
parser.add_argument('--cutoff', default=2500, type=int, help='EDS spectra cutoff')
parser.add_argument('--ignore_peaks', default=100, type=int, help='Number of initial peaks to ignore in spec2vec comparison')
parser.add_argument('--max_pairs', default=499500, type=int, help="Max number of pairs to calculate (default corresponds to all pairs of 1000 particles")
args = parser.parse_args()

# dont print warnings from matchms
from matchms import set_matchms_logger_level
set_matchms_logger_level("ERROR")

# create folder to store simelim output
args.sample=args.sample.replace(' ','_')
savepath=op.join(args.datadir, args.sample, 'simelim')
print('save path: ', savepath)
if not op.isdir(savepath):
    os.mkdir(savepath)

# load df     
df = pd.read_csv(op.join(args.datadir, args.sample, args.sample+'.csv'))
df = df.reset_index()

# get PSEM classes with greater than 10 particles
vc = df.PSEM_CLASS.value_counts()
choices = [x for x in vc.loc[vc>=10].index if x != 'Other']

for class_choice in choices:
    #choose PSEM CLASS to work with, limit to 5000 particles
    if len(df[df.PSEM_CLASS==class_choice]) > 5000:
        subset=df[df.PSEM_CLASS==class_choice].sample(n=5000, replace=False)
    else:
        subset=df[df.PSEM_CLASS==class_choice].copy()
    
    # load EDS into df
    datafile = h5py.File(op.join(args.datadir, args.sample, args.sample+'.hdf5'), 'r')
    subset['EDS'] = subset['PART#'].apply(lambda x: np.array(datafile[str(x)]['EDS']).flatten())
    
    # reset/redefine index
    subset['IMG']=subset['PART#'].apply(lambda x: f"./{x}.tif")    
    subset['PART#']=subset['index']
    
    print(f'Performing SimElim on class: {class_choice}')
    
    
    # Load data and apply filters
    spectrums = [simelim.spectrum_processing(s) for s in subset.apply(lambda x: simelim.Spect(x['EDS'], x, 
                                                                                              peak_cutoff=args.ignore_peaks, 
                                                                                              cutoff=args.cutoff), axis=1)]
    
    # Omit spectrums that didn't qualify for analysis
    spectrums = [simelim.spectrum_processing(s) for s in spectrums if s is not None]
    
    # Create spectrum documents
    reference_documents = [SpectrumDocument(s, n_decimals=2) for s in spectrums]
    
    #model training/importing
    model_file =  os.path.join(savepath, class_choice+'.model')
    if op.isfile(model_file):
        print(f'using pretrained model from {model_file}')
        model = gensim.models.Word2Vec.load(model_file)
    else:
        start_time = datetime.now()
        print('You are training a new word2vec model')
        model = train_new_word2vec_model(reference_documents, 
                                         iterations=30, 
                                         filename=model_file,
                                         workers=24, 
                                         progress_logger=True)
        end_time = datetime.now()
        print("training time:", end_time-start_time) 
    
        
    #load query data and apply filters
    query_spectrums = [s for s in subset.apply(lambda x: simelim.Spect(x['EDS'], x, 
                                                                       peak_cutoff=args.ignore_peaks,
                                                                       cutoff=args.cutoff), axis=1)]
    
    
    # Omit spectrums that didn't qualify for analysis
    query_spectrums = [s for s in query_spectrums if s is not None]
    
    # Create spectrum documents
    query_documents = [SpectrumDocument(s, n_decimals=2) for s in query_spectrums]
    
    # Define similarity_function
    spec2vec_similarity = Spec2Vec(model=model, 
                                   intensity_weighting_power=0.5,
                                   allowed_missing_percentage=30)
    
    
    #perform spec2vec
    pairs = list(itertools.combinations([SpectrumDocument(s, n_decimals=2) for s in spectrums], 2))
    
    if len(pairs)>args.max_pairs:
        random.shuffle(pairs)
        pairs = pairs[:args.max_pairs]
    
    print(f"{len(pairs)} pairs to compare")
    
    # create dataframe to store scores
    simfile = op.join(savepath, f'scores_{class_choice}.csv')
    pd.DataFrame({'reference' : [],'query' : [],'score' : []}).to_csv(simfile, mode='w', header=True, index=False)
    
    def sim_funct(pair, similarity=spec2vec_similarity, simfile=simfile):
        a,b = pair
        A = calculate_scores([a], [b], similarity, is_symmetric=True)
        for (reference,query,score) in A:
            pd.DataFrame(list(zip([reference.get('id')],
                                      [query.get('id')],
                                      [float(s) for s in score]))).to_csv(simfile, mode='a',header=False, index=False)
        return A
            
    with Pool(12) as p:
        simscore_output = p.map(sim_funct, pairs)
    
    scoredf=pd.read_csv(simfile)
    types_dict = {'reference': str, 'query': str,'score':float}
    for col, col_type in types_dict.items():
        scoredf[col] = scoredf[col].astype(col_type)    
        
    scoredf['x_l'] = scoredf['reference'].apply(lambda x: int(x.replace('spectrum','')))
    scoredf['x_r'] = scoredf['query'].apply(lambda x: int(x.replace('spectrum','')))
    scoredf['class_l'] = class_choice
    scoredf['class_r'] = class_choice
    scoredf.to_csv(simfile, index=False)
    print(f'scores saved to {simfile}')
        
