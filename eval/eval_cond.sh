if [ -z "$TYPE" ]; then
  echo "Error: 'CONDITION' variable is not set."
  exit 1
fi

# Eval
python -m eval.eval_condition \
    --eval_root ./outputs/${TYPE}_post \
    --gt_root ./data/organized_data/ \
    --list ./data/data_index/deduplicated_deepcad_testing_7_30.txt \
    --use_ray \
    --from_scratch \
    --num_cpus 24
