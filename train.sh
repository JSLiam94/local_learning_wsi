#!/bin/bash

# 检查是否传递了 GPU ID 参数
if [ -z "$1" ]; then
  echo "Usage: $0 <gpu_id>"
  exit 1
fi

gpu_id=$1

# 设置路径和参数
dataset_root="/home/zhengjingyuan/WSI2patch/ZDRN1.1"
#dataset_csv="/home/zhushenghao/data/JS/local_learning_wsi/alldata.csv"
dataset_csv="/home/zhushenghao/data/JS/local_learning_wsi/alldata_reduced_10.csv"
output_dir="/home/zhushenghao/data/JS/local_learning_wsi/run-new"

# 其他参数
num_workers=12
data_mean="0.6909,0.4654,0.6119"
data_std="0.1786,0.2102,0.1795"
precision=16
batch_size=1
accumulate_grad_batches=2
epochs=55
lr=1e-4
lr_factor=0.5
loss_weight="0.964218456,1.,1.,1.,1."
weight_decay=1e-2
K=4
alpha=1.
project_name="NB"
run_name="run-2"

# 调用 main.py
python main.py "$dataset_root" "$dataset_csv" \
  --num-workers "$num_workers" \
  --output-dir "$output_dir" \
  --precision "$precision" \
  --batch-size "$batch_size" \
  --accumulate-grad-batches "$accumulate_grad_batches" \
  --epochs "$epochs" \
  --lr "$lr" \
  --lr-factor "$lr_factor" \
  --loss-weight "$loss_weight" \
  --weight-decay "$weight_decay" \
  --decay-multi-epochs 25,35,45\
  --K "$K" \
  --alpha "$alpha" \
  --project-name "$project_name" \
  --run-name "$run_name" \
  --gpu-id "$gpu_id" \
  --num-classes 5\
  #--checkpoint_path "/home/zhushenghao/data/JS/local_learning_wsi/run/run-2/NB/u1kr9pif/checkpoints/epoch=8-step=176498.ckpt"