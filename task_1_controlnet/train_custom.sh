export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="./runs/controlnet_custom_dataset_15ksteps"   # output directory of each run

accelerate launch train.py \
--seed=42 \
--pretrained_model_name_or_path="$MODEL_DIR" \
--output_dir=$OUTPUT_DIR \
--dataset_name=liuch37/controlnet-cityscapes \
--resolution=512 \
--learning_rate=1e-5 \
--lr_scheduler "constant" \
--validation_image "./data/val_image_1.png" \
--validation_prompt "cars driving down a street in front of a futuristic, alien building, cars on fire" \
--train_batch_size=8 \
--gradient_accumulation_steps=1 \
--gradient_checkpointing \
--set_grads_to_none \
--use_8bit_adam \
--checkpoints_total_limit 3 \
--validation_steps 150 \
--report_to "tensorboard" \
--num_train_epochs 20 \
--caption_column "caption" \
--conditioning_image_column "seg" \
--max_train_steps 15000 \
# --resume_from_checkpoint "checkpoint-2000"

# liuch37/controlnet-cityscapes