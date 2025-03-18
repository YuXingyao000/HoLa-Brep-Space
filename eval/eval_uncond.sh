# Eval
python -m eval.sample_points --data_root ./outputs/unconditional_post --out_root ./outputs/unconditional_pcd --valid;
python -m eval.eval_brepgen --real ./data/organized_data --fake ./outputs/unconditional_pcd;
python -m eval.eval_complexity --eval_root ./outputs/unconditional_post --only_valid;
python -m eval.eval_condition \
    --eval_root ./outputs/unconditional_post \
    --gt_root ./data/organized_data/ \
    --list ./data/data_index/deduplicated_deepcad_testing_7_30.txt  \
    --num_cpus 24 \
    --use_ray \
    --from_scratch \
    --only_valid

# Validness
python -m eval.check_valid --data_root ./outputs/unconditional_post