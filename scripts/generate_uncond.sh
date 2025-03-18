CUDA_VISIBLE_DEVICES=0 python -m diffusion.train_diffusion \
  trainer.evaluate=true \
  trainer.batch_size=1000 \
  trainer.gpu=1 \
  trainer.test_output_dir=./outputs/unconditional/ \
  trainer.resume_from_checkpoint=./ckpt/Diffusion_uncond_1100k.ckpt \
  trainer.num_worker=2 \
  trainer.accelerator="32-true" \
  trainer.exp_name=test \
  dataset.name=Dummy_dataset \
  dataset.length=5000 \
  dataset.num_max_faces=30 \
  dataset.condition=None \
  model.name=Diffusion_condition \
  model.autoencoder_weights=./ckpt/AE_deepcad_1100k.ckpt \
  model.autoencoder=AutoEncoder_1119_light \
  model.with_intersection=true \
  model.in_channels=6 \
  model.dim_shape=768 \
  model.dim_latent=8 \
  model.gaussian_weights=1e-6 \
  model.pad_method=random \
  model.diffusion_latent=768 \
  model.diffusion_type=epsilon \
  model.gaussian_weights=1e-6 \
  model.condition=None \
  model.num_max_faces=30 \
  model.beta_schedule=linear \
  model.addition_tag=false \
  model.name=Diffusion_condition

python -m construct_brep \
    --data_root ./outputs/unconditional \
    --out_root ./outputs/unconditional_post \
    --use_ray \
    --num_cpus 24 \
    --drop_num 3 \
    --from_scratch