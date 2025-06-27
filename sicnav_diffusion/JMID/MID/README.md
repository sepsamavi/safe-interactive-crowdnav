
# Code

## Prepare Data

The data splits for the ETH/UCY (in meter), JRDB (in meter) datasets should be placed in ```raw_data```. We preprocess the data and generate .pkl files for training by running,

```
python process_data.py
```

You can specify a particular dataset with the --desired_sources argument.

The `train/validation/test/` splits are the same as those found in [Social GAN]( https://github.com/agrimgupta92/sgan). Please see ```process_data.py``` for details. For the JRDB dataset, we only process the train and validation sets since the authors of the dataset do not release the test set.

## Training
When you use JRDB dataset, Please note that you should change `JRDB_DEFAULTS_FILE` and `JRDB_CAMERA_CONFIGS_FILE` in `src/common.py`, corresponding to your JRDB dataset location.

### Step 1: Modify or create your own config file in ```/configs```

You can adjust parameters in config file as you like and change the network architecture of the diffusion model in ```models/diffusion.py```.

The following configs were used for the corresponding experiments:
* To recreate the JMID model used for the real-robot experiments in the RA-L paper,
  1. Train a JMID model on the JRDB dataset with a time interval of 0.25 seconds between frames for 90 epochs with an initial learning rate of 2x10^{−4} and an exponential decay learning rate scheduler with a decay factor of 0.98 (the exponential decay is already in the code and does not need to be specified in the config).
  2. Continue training the JMID model on the JRDB dataset for another 25 epochs with  ddim_jp_jrdb_bs64_lr0000325_ep5000_finetuning_lr0002_model.yaml. The original checkpoint is in checkpoints/jmid/jrdb_bev_0_25_multi_class_epoch25.pt
  3. Finetune the JMID model on data that was collected in the Vicon room used for experiments for 35 epochs with ddim_jp_p3_bs64_lr00002_SICNav_structured_subset.yaml. The original checkpoint is in /checkpoints/jmid/SICNav_TRO_MID_data_3m_structured_subset_epoch35.pt
* To recreate the iMID model used for the real-robot experiments,
  1. Train an iMID model on the JRDB dataset with a time interval of 0.25 seconds between frames for 100 epochs with ddim_p3_bs256_lr001_jrdb_bev_0_25_multi_class_clean.yaml. The original checkpoint is in /checkpoints/imid/jrdb_bev_0_25_multi_class_clean_epoch100.pt
  2. Finetune the iMID model on data collected in the Vicon room used for experiments for 20 epochs with ddim_p3_bs256_lr0002_epochs100_SICNav_structured_subset.yaml. The original checkpoint is here: /checkpoints/imid/SICNav_TRO_MID_data_3m_structured_subset_epoch20.pt
* To recreate the JMID models evaluated on the ETH/UCY benchmark, train a JMID model on the appropriate splits for 500 epochs with ddim_jp_p3_bs64_lr0001_X.yaml, where "X" is the split name. The original checkpoints are here: /checkpoints/jmid/
* To recreate the iMID models evaluated on the ETH/UCY benchmark, train an iMID model on the appropriate splits for 900 epochs with ddim_p3_bs256_lr001_X.yaml, where "X" is the split name. The original checkpoints are here: /checkpoints/imid/
* To recreate the JMID model evaluated on the JRDB dataset, train a JMID model on the JRDB dataset with a time interval of 0.4 seconds between frames for 200 epochs with ddim_jp_p3_b3_r3_bs64_lr0001_jrdb_bev_0_4_multi_class_clean.yaml. The original checkpoints are here: /checkpoints/jmid/
* To recreate the iMID model evaluated on the JRDB datset, train an iMID model on the JRDB dataset with a time interval of 0.4 seconds between frames for 450 epochs with ddim_p3_bs256_lr001_jrdb_bev_0_4_multi_class_clean.yaml. The original checkpoints are here: /checkpoints/imid/

Make sure the ```eval_mode``` is set to False.

Note: Some of the configs in the ```/configs``` directory were used with a previous commit of the code so may not work with the current state of the code.

### Step 2: Train model (MID, JMID)

 ```python main.py --config configs/YOUR_CONFIG.yaml```

 Note that the dataset used for training and evaluation are specified in the config file.

Logs and checkpoints will be automatically saved.

Configs:
* ddim_jp.yaml: For training JMID on JRDB dataset with 0.4 dt
* ddim_jp_p3_bs64_lr0001_x.yaml: For JMID training on x
* ddim_p3_bs256_lr001_x.yaml: For training iMID on x
* ddim_p3_bs256_lr001_jrdb_bev_0_4_multi_class_clean.yaml: For training iMID on JRDB

## Evaluation

To evaluate a trained-model, please set ```eval_mode``` in config file to True and set the epoch you'd like to evaluate at from ```eval_at``` and run

 ```python main.py --config configs/YOUR_CONFIG.yaml --dataset DATASET```

 You can evaluate with DDPM or DDIM sampling.

# References
This code base in this folder was developed by Anthony Lem and Fumiaki Sato.

This code base initiated as a fork of the code for CVPR 2022 paper "Stochastic Trajectory Prediction via Motion Indeterminacy Diffusion" [[Code]](https://github.com/Gutianpei/MID) by Tianpei Gu*, Guangyi Chen*, Junlong Li, Chunze Lin, Yongming Rao, Jie Zhou and Jiwen Lu, released under the MIT license.

### License
Our code is released under MIT License.
