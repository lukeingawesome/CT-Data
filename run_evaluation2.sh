# =============================================================================
# VALIDATION SPLIT EVALUATION (to match training validation exactly)
# =============================================================================

# 1-Channel Model - VALIDATION SPLIT
CUDA_VISIBLE_DEVICES=5 python3 evaluate_retrieval.py \
    --checkpoint /model/1c_siglip_orig/pytorch_model.bin \
    --csv_file csv/all_ct.csv \
    --output_dir ./results_1c_val \
    --dataset_mode ct \
    --batch_size 10 \
    --device cuda \
    --img_column "img_path" \
    --text_column "findings" \
    --text_separator "!@#$%^&*()" \
    --precision "bf16" \
    --split "train" \
    --split_column "split" \
    --text_base "microsoft/LLM2CLIP-Llama-3.2-1B-Instruct-CC-Finetuned"

# # 3-Channel Model - VALIDATION SPLIT  
# CUDA_VISIBLE_DEVICES=5 python3 evaluate_retrieval.py \
#     --checkpoint /model/3c_siglip_orig/pytorch_model.bin \
#     --csv_file csv/all_ct.csv \
#     --output_dir ./results_3c_val \
#     --dataset_mode ct \
#     --batch_size 10 \
#     --device cuda \
#     --img_column "img_path" \
#     --text_column "findings" \
#     --text_separator "!@#$%^&*()" \
#     --precision "bf16" \
#     --split "val" \
#     --split_column "split" \
#     --use_3channel \
#     --text_base "microsoft/LLM2CLIP-Llama-3.2-1B-Instruct-CC-Finetuned"

# =============================================================================
# Quick Usage Examples
# =============================================================================

# Example 1: 1-channel model with your CSV
# CUDA_VISIBLE_DEVICES=4 python3 evaluate_retrieval.py --checkpoint /model/1c_siglip/pytorch_model.bin --csv_file /your/data.csv --output_dir ./results_1c --dataset_mode ct --batch_size 10

# Example 2: 3-channel model with your CSV  
# CUDA_VISIBLE_DEVICES=4 python3 evaluate_retrieval.py --checkpoint /model/3c_siglip/pytorch_model.bin --csv_file /your/data.csv --output_dir ./results_3c --dataset_mode ct --use_3channel --batch_size 10

# Example 3: Different batch size
# CUDA_VISIBLE_DEVICES=4 python3 evaluate_retrieval.py --checkpoint /model/1c_siglip/pytorch_model.bin --csv_file /your/data.csv --output_dir ./results --batch_size 16 --dataset_mode ct

# Example 4: Custom column names
# CUDA_VISIBLE_DEVICES=4 python3 evaluate_retrieval.py --checkpoint /model/1c_siglip/pytorch_model.bin --csv_file /your/data.csv --output_dir ./results --img_column "image_path" --text_column "findings" --dataset_mode ct --batch_size 10