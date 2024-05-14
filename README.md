# Few-shot Learning and Similarity Graph Clustering for Automated Particle Analysis

## Collect data
The automated particle analysis (SEM/EDS) dataset (Ritchie & Renolds, 2021) used in this study is openly available on the NIST website (https://data.nist.gov/od/id/mds2-2476). Note that downloaded files should be unzipped and spaces in directory names should be converted to underscores prior to running the following scripts. The examples below assume the collected data is in separate folders labeled by sample name in ./data/raw/.

## Preprocess data
The raw data from each sample should be converted to an hdf5 database prior to training or inference. The example below demonstrates how to generate the hdf5 and corresponding csv with particle info for a sample.

```
export SAMPLE='Shooter_#3_-_Zero_time_R'
python preprocess.py --datadir ./data/raw/ --savedir ./data/processed/ --sample ${SAMPLE}
```

###  Spectral similarity [Optional]
Particle pairs of the same class can be filtered by spectral similarity if desired. Note that spec2vec and matchms are required packages for this step.

```
export SAMPLE='Shooter_#3_-_Zero_time_R'
python simelim.py --datadir ./data/processed/ --sample ${SAMPLE} --ignore_peaks 100
```

## Train a model
The command-line arguments below will reproduce training of the multi-modal few-shot network. Remove the --sem flag to reproduce the single modality (EDS) few-shot network. Note that stiochasicity during training will lead to slight differences in final model performance.

```
python train.py --model_name 'multimodal_model' \
 --database ./data/processed/Shooter_#3_-_Zero_time_R/Shooter_#3_-_Zero_time_R.hdf5 \
            ./data/processed/Shooter_#3_-_Zero_time_L/Shooter_#3_-_Zero_time_L.hdf5 \
            ./data/processed/Ford_Explorer_A213_Front_Driver/Ford_Explorer_A213_Front_Driver.hdf5 \
            ./data/processed/Ford_Explorer_A213_Front_Passenger/Ford_Explorer_A213_Front_Passenger.hdf5 \
            ./data/processed/Ford_Explorer_A213_Rear_Driver/Ford_Explorer_A213_Rear_Driver.hdf5 \
            ./data/processed/Ford_Explorer_A213_Rear_Passenger/Ford_Explorer_A213_Rear_Passenger.hdf5 \
            ./data/processed/Spinners_-_Debris_from_spinner/Spinners_-_Debris_from_spinner.hdf5 \
            ./data/processed/Spinners_-_Post-cleanup/Spinners_-_Post-cleanup.hdf5 \
            ./data/processed/Spinners_-_Post-handling,_pre-ignition/Spinners_-_Post-handling,_pre-ignition.hdf5 \
            ./data/processed/Spinners_-_Post-ignition/Spinners_-_Post-ignition.hdf5 \
 --savedir './models/' \
 --max_class_sample 500 \
 --prepool --pool_size 3 \
 --epochs 50 --batch 128 --parallel \
 --simelim --simelim_percentile 2.2 \
 --sem
```

## Few-shot similarity
Pairwise similarity scores for a new sample can be obtained as follows. Note that the sample must be preprocessed, but spectral similarity scores are not needed.

```
export SAMPLE='Shooter_#1_-_Zero_time'
export TAG='subsample1'
python inference.py --modeldir ./models/multimodal_model/ \
 --database ./data/processed/${SAMPLE}/${SAMPLE}.hdf5 \
 --tag ${TAG} --n-samples 10 --n-particles 500
```

## Graph metrics
The similarity graph and associated graph metrics can be obtained for the new sample as follows.

```
export SAMPLE='Shooter_#1_-_Zero_time'
export TAG='subsample1'
python metrics.py --edgefile ./models/multimodal_model/inference/${SAMPLE}/${SAMPLE}_${TAG}_edgelist_mean.csv --savefile ./models/multimodal_model/inference/${SAMPLE}/${SAMPLE}_${TAG}_graph --metrics 'leadership' 'bonding' 'diversity' 'class_assortativity' 'degree_assortativity' 'p_smoothness' 'algebraic_connectivity' --data ./data/processed/${SAMPLE}/${SAMPLE}.csv
```



