config="cfgs/nuscenes_models/E4/DSP/cbgs_pillar0075_neckv1_res2d_centerpoint_ds.yaml"
arg='E4-DSP_concat_scatter_mean_no_attention'
ckpt="../output/nuscenes_models/E4/DSP/cbgs_pillar0075_neckv1_res2d_centerpoint_ds/${arg}/ckpt/checkpoint_epoch_20.pth"

# train
./scripts/dist_train.sh 4 --cfg_file ${config} --extra_tag ${arg} --fix_random_seed --workers 2 --tcp_port 10000 #--find_unused_parameters

# inference speed
python3 test.py --cfg_file ${config} --ckpt ${ckpt} --infer_time --batch_size 16 --workers 0 --extra_tag ${arg}

# test
#./scripts/dist_test.sh 4 --cfg_file ${config} --ckpt ${ckpt}
