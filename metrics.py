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

import networkx as nx
import time
import pandas as pd
import numpy as np
import os
import os.path as op
import multiprocessing
from utils import graphmetrics
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--edgefile', type=str, required=True, help='path to edgelist')
parser.add_argument('--savefile', type=str, required=True, help='path to save output')
parser.add_argument('--metrics', required=True, nargs='+', type=str, help='list of metrics to compute')
parser.add_argument('--n-int', type=int, default=10, help='Number of evenly spaced points between 0 and 1 for thresholding.')
parser.add_argument('--data', type=str, default='', help='Path to .csv of particle data (required for class_assortativity).')
args = parser.parse_args()

start=time.time()

oldscores={'metrics':{}}
newscores={'metrics':{}}

# set up functions to calculate
funcs = args.metrics
func_map = {'leadership':graphmetrics.leadership, 
            'bonding':graphmetrics.bonding, 
            'diversity':graphmetrics.diversity,  
            'simple_leadership':graphmetrics.simple_leadership,
            'skewness_leadership':graphmetrics.skewness_leadership,
            'distance':graphmetrics.average_distance, 
            'class_assortativity':graphmetrics.class_assortativity,
            'degree_assortativity':graphmetrics.degree_assortativity,
            'components':graphmetrics.components,
            'component_max':graphmetrics.component_max, 
            'component_min':graphmetrics.component_min,
            'component_mean':graphmetrics.component_mean,
            'p_smoothness':graphmetrics.p_smoothness,
            'algebraic_connectivity':graphmetrics.algebraic_connectivity,
            }

# check that save file is .json
if not args.savefile.endswith('.json'):
    args.savefile=args.savefile+'.json'

# make sure save directory exists
if '/' in args.savefile:
    savedir='/'.join(args.savefile.split('/')[:-1])
    if not op.isdir(savedir):
        try:
            os.mkdir(savedir)
        except:
            print(f"savedir already present: {op.isdir(savedir)}")
            if not op.isdir(savedir):
                os.mkdir(savedir)

g=graphmetrics.simG()
if op.isfile(args.savefile):
    # load data
    g.load(args.savefile)
    # extract previously computed metrics
    if 'metrics' in g.data.keys():
        oldscores['metrics']=g.data['metrics']   
else:
    # make new graph
    g.make(args.edgefile)
    
if 'class_assortativity' in args.metrics:
    if args.data == '':
        print(f"path to processed data missing (--data flag); removing class_assortivity from metric calculation")
        funcs.remove('class_assortativity')
    else:
        sample = pd.read_csv(args.data)
        PSEM_class={n:sample.loc[sample['PART#']==int(n)]["PSEM_CLASS"].tolist()[0] for n in g.G.nodes()}
        nx.set_node_attributes(g.G, PSEM_class, name="PSEM_CLASS")

# confirm G is a complete graph
if g.complete:

    # get threshold information
    if 'thresholds' in oldscores['metrics'] and str(args.n_int) in oldscores['metrics']['thresholds'].keys():
        thresholds = oldscores['metrics']['thresholds'][str(args.n_int)]
    else:
        # determine thresholds
        if args.n_int:
            thresholds=[y/args.n_int for y in range(args.n_int+1)]
        else:
            thresholds=sorted(list(set([g.G.edges[e]['weight'] for e in g.G.edges()]))) 

        #needed to compute interval widths
        if 0 not in thresholds: 
            thresholds=[0]+thresholds
        if 1 not in thresholds: 
            thresholds=thresholds+[1]
            
        newscores=graphmetrics.merge(newscores, {'metrics':{'thresholds':{str(args.n_int): thresholds}}})

    # get width information
    idxes = list(range(len(thresholds)-1))
    if 'widths' in oldscores['metrics'] and str(args.n_int) in oldscores['metrics']['widths'].keys():
        widths = oldscores['metrics']['widths'][str(args.n_int)]
    else:
        # get interval widths
        widths = [thresholds[i+1]-thresholds[i] for i in idxes]
        newscores=graphmetrics.merge(newscores, {'metrics':{'widths':{str(args.n_int): widths}}})

    # get thresholded graphs
    TGs = [graphmetrics.thresh(g.G, thresholds[i+1], isolates=True) for i in idxes]

    for func in funcs:
        # check that function wasn't computed previously
        if func in oldscores['metrics'] and str(args.n_int) in oldscores['metrics'][func].keys():
            funcs.remove(func)
            print(f"{func} already calculated for {args.n_int}...skipping")
            
    for func in funcs:
        f = func_map[func]
        pool = multiprocessing.Pool()
        results = pool.map(f, TGs)
        pool.close()
        pool.join()
        newscores=graphmetrics.merge(newscores, {'metrics':{func: {str(args.n_int): {'overall':sum([w*v for w,v in zip(widths, results)]),
                                                                                     'thresh':results}}}})
    g.add_to_data(graphmetrics.merge(oldscores, newscores), 'metrics')
    g.write(args.savefile)

    print(f"results saved to {args.savefile}: {g.complete}")
else:
    print(f"G is not a complete graph; metrics not computed")
print(f"finished after {time.time()-start:0.1f} s")