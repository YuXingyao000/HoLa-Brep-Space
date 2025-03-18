if [ -z "$TYPE" ]; then
  echo "Error: 'CONDITION' variable is not set."
  exit 1
fi

python -m construct_brep \
    --data_root ./outputs/${TYPE} \
    --out_root ./outputs/${TYPE}_post \
    --use_ray \
    --num_cpus 2 \
    --drop_num 3 \
    --from_scratch