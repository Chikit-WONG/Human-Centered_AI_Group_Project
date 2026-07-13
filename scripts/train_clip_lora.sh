export CUDA_VISIBLE_DEVICES=0 # we only use a single gpu to train clip alignment

MODEL_ROOT="/home/jiawen/pretrained"
MODEL_ID="laion/CLIP-ViT-B-32-laion2B-s34B-b79K" # "laion/CLIP-ViT-H-14-laion2B-s32B-b79K" "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
MODEL_NAME="${MODEL_ID//\//-}"
MODEL_PATH="${MODEL_ROOT}/${MODEL_ID}"
IMAGE_DIR="/home/jiawen/data/things-eeg"
BRAIN_DIR="/home/jiawen/data/things-eeg/Preprocessed_data_250Hz_whiten"
SUBJECT_IDS=(1 2 3 4 5 6 7 8 9 10)
DATASET=things
BRAIN_COLUMN=eeg
TIME="0,250"
AVG_TRIALS=true
CHANNELS="P7,P5,P3,P1,Pz,P2,P4,P6,P8,PO7,PO3,POz,PO4,PO8,O1,Oz,O2"
LR=5.0e-4
VLR=5.0e-5
SEED=42
EPOCHS=25
BACKBONE=brain_mlp
LORA_RANK=32
LORA_LAYERS=all-linear

for subj in "${SUBJECT_IDS[@]}"; do
    run_name="clip-${DATASET}-${BRAIN_COLUMN}-subj${subj}-${MODEL_NAME}-r${LORA_RANK}"
    out_dir="/home/jiawen/exp/${run_name}"
    torchrun train_clip_lora.py \
        --dataset_name ${DATASET} --brain_directory ${BRAIN_DIR} --image_directory ${IMAGE_DIR} \
        --subject_ids ${subj} --brain_column ${BRAIN_COLUMN} \
        --brain_backbone ${BACKBONE} --dropout 0.1 --pretrained_model_name_or_path ${MODEL_PATH} \
        --lora_rank ${LORA_RANK} --lora_layers ${LORA_LAYERS} --gradient_checkpointing \
        --time_slice ${TIME} --avg_trials --selected_channels ${CHANNELS} \
        --learning_rate ${LR} --vision_learning_rate ${VLR}  --lr_scheduler_type cosine --weight_decay 0.05 \
        --seed ${SEED} --dataloader_num_workers 8 --mixed_precision bf16 \
        --report_to swanlab --output_dir ${out_dir} --run_name ${run_name} --save_total_limit 1 \
        --num_train_epochs ${EPOCHS} --per_device_train_batch_size 512  --per_device_eval_batch_size 100 

done




