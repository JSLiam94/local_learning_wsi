#!/bin/bash

# 检查是否传递了 GPU ID 参数
if [ -z "$1" ]; then
  echo "Usage: $0 <gpu_id>"
  exit 1
fi

gpu_id=$1

# 设置路径和参数
dataset_root="/home/zhushenghao/data/JS/nb"
dataset_csv="/home/zhushenghao/data/JS/local_learning_wsi/output.csv"
output_dir="/home/zhushenghao/data/JS/local_learning_wsi/run"
checkpoint_path="/home/zhushenghao/data/JS/local_learning_wsi/run/run-2/NB/u1kr9pif/checkpoints/epoch=8-step=176498.ckpt" # 模型检查点路径

# 其他参数
num_workers=4
data_mean="0.6909,0.4654,0.6119"
data_std="0.1786,0.2102,0.1795"
precision=32
batch_size=4
loss_weight="0.964218456,1.,1.,1.,1."
weight_decay=1e-2
K=8
alpha=1.
project_name="NB"
run_name="run-2"
num_classes=5 # 确保这与训练时使用的类别数一致

# 调用 main.py 进行评估
python eval.py "$dataset_root" "$dataset_csv" \
  --num-workers "$num_workers" \
  --output-dir "$output_dir" \
  --precision "$precision" \
  --batch-size "$batch_size" \
  --loss-weight "$loss_weight" \
  --weight-decay "$weight_decay" \
  --K "$K" \
  --alpha "$alpha" \
  --project-name "$project_name" \
  --run-name "$run_name" \
  --gpu-id "$gpu_id" \
  --num-classes "$num_classes" \
  --checkpoint_path "/home/zhushenghao/data/JS/local_learning_wsi/run/run-2/NB/u1kr9pif/checkpoints/epoch=8-step=176498.ckpt"