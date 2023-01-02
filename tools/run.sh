config="cfgs/nuscenes_models/cbgs_pillar0075_neckv1_res2d_centerpoint_exp.yaml"
arg='z_axis_error_fixed_1,2/transformer_decoder_bin32_relu'
ckpt="../output/nuscenes_models/cbgs_pillar0075_neckv1_res2d_centerpoint_exp/${arg}/ckpt/checkpoint_epoch_20.pth"

# train
#./scripts/dist_train.sh 4 --cfg_file ${config} --extra_tag ${arg} --fix_random_seed --workers 2 --tcp_port 10000

# test
#./scripts/dist_test.sh 4 --cfg_file ${config} --ckpt ${ckpt}

# inference speed
python3 test.py --cfg_file ${config} --ckpt ${ckpt} --infer_time --batch_size 1

