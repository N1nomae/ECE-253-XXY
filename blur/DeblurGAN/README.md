# DeblurGAN
[arXiv Paper Version](https://arxiv.org/pdf/1711.07064.pdf)

Pytorch implementation of the paper DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks. Borrowed and adopted from the original git repo for DeblurGAN.

### 1) Put images
Place blurry images in:
- `./my_blur_images/`

### 2) Put weights
Make sure the generator checkpoint exists at:
- `./checkpoints/experiment_name/latest_net_G.pth`

### 3) Run deblurring
python test.py --dataroot ./my_blur_images --model test --dataset_mode single --learn_residual --name experiment_name --display_id 0 --resize_or_crop none

### 4) Outputs
Deblurred images are saved to:
- `./results/experiment_name/deblurred/`
