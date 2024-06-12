export MASTER_PORT=88888

echo $CUDA_VISIBLE_DEVICES
nvidia-smi
# End visible GPUs.

which python

python imagenet/main.py -a resnet18 /mnt/hdd/hsyoon/data/imagenet-100 \
    --experiment_name 'sbece' \
    --seed 52 \
    --loss 'ce+sbece' \
    --calibration \
    --calset_ratio 0.1 \
    --T 0.01 \
    --lamda 2.0 \