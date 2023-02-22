config="cfgs/nuscenes_models/E4/SST/cbgs_pillar0075_neckv1_res2d_centerpoint_sst.yaml"
<<<<<<< HEAD
arg='E4-sst'
ckpt="../output/nuscenes_models/E4/SST/cbgs_pillar0075_neckv1_res2d_centerpoint_sst/${arg}/ckpt/checkpoint_epoch_20.pth"
=======
arg='E4-sst_nhead_4'
ckpt="../output/nuscenes_models/E4/ZBAM/cbgs_pillar0075_neckv1_res2d_centerpoint_mlp_zbam/${arg}/ckpt/latest_model.pth"
>>>>>>> 8c3b05ccd0da5a8bfb2f66ae3fb9cf51cf654457

# train
#./scripts/dist_train.sh 4 --cfg_file ${config} --extra_tag ${arg} --fix_random_seed --workers 2 --tcp_port 10000 #--find_unused_parameters

# inference speed
python3 test.py --cfg_file ${config} --ckpt ${ckpt} --infer_time --batch_size 16 --workers 0 --extra_tag ${arg}

# test
#./scripts/dist_test.sh 4 --cfg_file ${config} --ckpt ${ckpt}
