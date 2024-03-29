exp_name = xxx
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; python -m torch.distributed.launch --nproc_per_node 8 run_logicnlg.py \
  --do_train \
  --do_eval \
  --dataset_name kasnerz/logicnlg \
  --model_name_or_path Yale-LILY/reastap-large \
  --overwrite_output_dir \
  --output_dir checkpoints/logicnlg/${exp_name} \
  --max_source_length 1024 \
  --max_target_length 128 \
  --val_max_target_length 128 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --per_device_eval_batch_size 16 \
  --report_to wandb \
  --num_train_epochs 20 \
  --warmup_ratio 0.1 \
  --learning_rate 2e-5 \
  --fp16 \
  --logging_steps 10 \
  --eval_steps 200 \
  --save_steps 400 \
  --evaluation_strategy steps \
  --predict_with_generate \
  --num_beams 5 \
  --weight_decay 1e-2 \
  --label_smoothing_factor 0.1 \
  --generation_max_length 128 \
  --save_total_limit 8 \
  --run_name ${exp_name}