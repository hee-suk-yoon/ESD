export MASTER_PORT=88888

echo $CUDA_VISIBLE_DEVICES
nvidia-smi
# End visible GPUs.

which python

python imagenet/main.py -a resnet18 /mnt/hdd/hsyoon/data/imagenet-100 \
    --experiment_name 'esd' \
    --seed 52 \
    --loss 'ce+esd' \
    --calibration \
    --calset_ratio 0.1 \
    --lamda 4.0 \