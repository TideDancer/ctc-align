export EPOCH=1
export BATCH_SIZE=4
export BATCH_ACC=8
export LR=1e-4

export CTC_TYPE=$1 # standard, no-skip
export PRIOR_TYPE=$2 # no, entropy, uniform, posterior
export VOCAB_TYPE=$3 # regular, splitter, s-boundary, t-boundary 

CUDA_VISIBLE_DEVICES=0 python run_pr.py \
--output_dir ./output/test \
--overwrite_output_dir \
--ctc_type $CTC_TYPE \
--prior_type $PRIOR_TYPE \
--vocab $VOCAB_TYPE \
--num_train_epochs $EPOCH \
--per_device_train_batch_size $BATCH_SIZE \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps $BATCH_ACC \
--evaluation_strategy="steps" \
--warmup_ratio 0.1 \
--save_steps="500" \
--eval_steps="500" \
--do_train \
--do_eval \
--logging_steps="100" \
--learning_rate $LR \
--model_name_or_path="facebook/wav2vec2-base" \
--save_total_limit 1 \
--fp16 \
--dataset_name="timit_asr" \
--train_split_name="train" \
--validation_split_name="test" \
--orthography="timit" \
--preprocessing_num_workers=20 \
--freeze_feature_extractor \
--verbose_logging \
--group_by_length

