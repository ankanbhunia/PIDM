
### DDIM Sampling (Faster)
```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port 48109  utils/gen.py \ 
        --exp_name="pidm_deepfashion" \
        --checkpoint_name=last.pt \
        --dataset_path "./dataset/deepfashion/"  \
        --sample_algorithm ddim
```
### DDPM Sampling
```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port 48109  utils/gen.py \ 
        --exp_name="pidm_deepfashion" \
        --checkpoint_name=last.pt \
        --dataset_path "./dataset/deepfashion/"  \
        --sample_algorithm ddpm
```

Output images are saved inside ```images``` folder. 

### Folder structure for checkpoint files
```
PIDM/
  checkpoints/
    <exp_name>/
      <checkpoint_name.pt>
```
