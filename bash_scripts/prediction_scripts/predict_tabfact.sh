export CUDA_VISIBLE_DEVICES=0; python run_tabfact.py \
  --do_predict \
  --model_name_or_path Yale-LILY/reastap-large-finetuned-tabfact \
  --overwrite_output_dir \
  --output_dir outputs/tabfact_output \
  --per_device_eval_batch_size 32 \
  --eval_accumulation_steps 32 \
  --inference_set test
