if [ -z "$TYPE" ]; then
  echo "Error: 'CONDITION' variable is not set."
  exit 1
fi

cd ./eval/lfd/evaluation_scripts/compute_lfd_feat
python -m compute_lfd_feat_multiprocess --gen_path ../../../../outputs/${TYPE}_post --save_path ../../../../outputs/${TYPE}_lfd_feat --prefix recon_brep.stl 
cd ..
python -m compute_lfd --dataset_path ../../../data/data_lfd_feat --gen_path ../../../outputs/${TYPE}_lfd_feat --save_name ../../../outputs/${TYPE}_lfd.pkl --num_workers 8 --list ../../../data/data_index/deduplicated_deepcad_training_7_30.txt 
cd ../../..
python -m eval.viz_lfd ./outputs/${TYPE}_lfd.pkl ./outputs/${TYPE}_lfd.png ./outputs/${TYPE}_post