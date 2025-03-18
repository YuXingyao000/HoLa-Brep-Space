if [ -z "$TYPE" ]; then
  echo "Error: 'TYPE' variable is not set."
  exit 1
fi
python -m sort_and_merge \
    --data_root ./outputs/${TYPE}_post \
    --out_root ./outputs/${TYPE}_post_sorted \
    --sort \
    --valid \
    --index \
    --use_ray