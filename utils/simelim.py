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
import numpy as np
import pandas as pd
import matchms.filtering as msfilters
from matchms import Spectrum

def spectrum_processing(s, peak_cutoff=100, cutoff=2500):
    """Build spectrum processing pipeline.

    Args:
      s: matchms.Spectrum class.
      peak_cutoff: Index before which spectrum is trimmed.
      cutoff: Index after which spectrum is trimmed.

    Returns:
      Pipeline function.
    """     
    
    s = msfilters.default_filters(s)
    s = msfilters.add_parent_mass(s, estimate_from_adduct=False)
    s = msfilters.normalize_intensities(s)
    s = msfilters.reduce_to_number_of_peaks(s, n_required=10, ratio_desired=None, n_max=cutoff-peak_cutoff)
    s = msfilters.select_by_mz(s, mz_from=0, mz_to=1000)
    s = msfilters.add_losses(s, loss_mz_from=10.0, loss_mz_to=200.0)
    s = msfilters.require_minimum_number_of_peaks(s, n_required=10)
    return s

def Spect(x, row, peak_cutoff=100, cutoff=2500):
    """Build spectrum processing pipeline.

    Args:
      x: Spectrum.
      rowname: Row in dataframe to which spectrum belongs.
      peak_cutoff: Index before which spectrum is trimmed.
      cutoff: Index after which spectrum is trimmed.

    Returns:
      matchms.Spectrum class.
    """    
    # cut off first x peaks to remove common elements (oxygen, carbon, etc) 
    x=x[peak_cutoff:] 
    return Spectrum(mz=np.arange(peak_cutoff, cutoff, dtype=np.float64),
                    intensities=np.reshape(x, (cutoff-peak_cutoff,)),
                    metadata={'id': f"spectrum{row.IMG.split('/')[-1].split('.')[0]}"}) 

def make_simelim_pairs(args, cutoff_score=0.99):
    """Build spectrum processing pipeline.

    Args:
      args: Command-line arguments.
      cutoff_score: Value at or above which spectra are called similar.

    Returns:
      Dataframe of pairwise simelim scores.
    """     
    # collect all simelim score files
    score_files = [x for x in os.listdir(op.join(args.savedir, 'simelim')) if x.endswith('.csv')]

    # merge all into a single dataframe
    sf = pd.DataFrame()
    for fil in score_files:
        tmp = pd.read_csv(op.join(args.savedir, 'simelim', fil))
        sf = pd.concat([sf,tmp], ignore_index=True, sort=False)

    return sf.loc[sf.score>=cutoff_score].copy()
