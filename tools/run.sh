config="cfgs/nuscenes_models/cbgs_pillar0075_neckv1_res2dE2_centerpoint_mlp.yaml"
arg='E2_Transformer'
ckpt="../output/nuscenes_models/cbgs_pillar0075_neckv1_res2dE2_centerpoint_transformer/${arg}/ckpt/checkpoint_epoch_20.pth"
ckpt="../E2_MLP.pth"
# train
#./scripts/dist_train.sh 4 --cfg_file ${config} --extra_tag ${arg} --fix_random_seed --workers 2 --tcp_port 10000

# test
#./scripts/dist_test.sh 4 --cfg_file ${config} --ckpt ${ckpt}

# inference speed
python3 test.py --cfg_file ${config} --ckpt ${ckpt} --infer_time --batch_size 16 --workers 0 --extra_tag ${arg}

