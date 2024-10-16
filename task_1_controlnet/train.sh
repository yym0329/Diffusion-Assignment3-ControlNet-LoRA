export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="./runs/controlnet_fill50k_50k_lrlinear_3epochs_batch2"   # output directory of each run

accelerate launch train.py \
--seed=42 \
--pretrained_model_name_or_path=$MODEL_DIR \
--output_dir=$OUTPUT_DIR \
--dataset_name=fusing/fill50k \
--resolution=512 \
--learning_rate=1e-5 \
--lr_scheduler "linear" \
--validation_image "./data/conditioning_image_1.png" "./data/conditioning_image_2.png" \
--validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
--train_batch_size=2 \
--gradient_accumulation_steps=4 \
--gradient_checkpointing \
--set_grads_to_none \
--use_8bit_adam \
--checkpoints_total_limit 2 \
--validation_steps 100 \
--report_to "tensorboard" \
--num_train_epochs 4 \
# --resume_from_checkpoint "task_1_controlnet/runs/controlnet_fill50k_50k_lr1e-5_3epochs/checkpoint-500" 
