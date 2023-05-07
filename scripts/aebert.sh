MODEL=${1:-roberta-base}
DATASEED=${2:-7}
DATANUM=${3:-100}
TASKNAME=${4:-imdb}
NUMSETS=${5:-1}
VAESETS=${6:-128,256,512}
LAYERSETS=${7:-6,8,11}
GPU=${8:-0}
SEED=${9:-7}
RLW=${10:-0.001}
DROPOUT=${11:-0.1}
MIXOUT=${12:-0.1}

export CUDA_VISIBLE_DEVICES=$GPU
export USE_ADAPTER=AE2TK

LR=2e-5
EPOCH=100
# EPOCH=25
REPO=$PWD
OUT_DIR="$REPO/outputs/"

SAVE_DIR="${OUT_DIR}/${TASKNAME}/${MODEL}.${USE_ADAPTER}${RLW}.${NUMSETS}.${VAESETS}.${LAYERSETS}.${DROPOUT}.${MIXOUT}-LR${LR}-epoch${EPOCH}-${DATASEED}-${DATANUM}-${SEED}"
mkdir -p $SAVE_DIR

python src/run_ae.py  --model_name_or_path  $MODEL  \
    --output_dir $SAVE_DIR  --task_name $TASKNAME --model_type bert  --do_eval\
    --max_seq_length 128  --num_train_epochs $EPOCH   --overwrite_output_dir \
    --outputfile results/results.csv  --do_lower_case  --ib_dim 384 \
    --beta 1e-05 --ib --learning_rate $LR --recon_loss_weight $RLW --do_train --num_samples $DATANUM\
    --eval_types dev train test   --kl_annealing linear --evaluate_after_each_epoch --seed $SEED\
    --data_seed $DATASEED --num_sets $NUMSETS --vae_sets $VAESETS --layer_sets $LAYERSETS\
    --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 2560 --dropout $DROPOUT --mixout $MIXOUT #--sample_train 
