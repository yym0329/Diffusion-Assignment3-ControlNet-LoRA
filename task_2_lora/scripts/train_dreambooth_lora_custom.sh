export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="./sample_data/dreambooth-jjh-small"
export OUTPUT_DIR="./runs/dreambooth-jjh-small"

accelerate launch --mixed_precision="no" train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of a sks man" \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=500 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=700 \
  --validation_prompt="a sks man walking" \
  --validation_epochs=50 \
  --checkpoints_total_limit 2 \
  --seed="0"