# WGAN-with-Cross-Attention
Project to get attention from discriminator: 1st modification


##Modifications to the original WGAN implementation:
1. Added feedback from the discriminator in the form of attention


##Execute
python main.py --dataset lsun --dataroot /workspace/lsun --cuda --clamp --fdbac--save_dir samples_no_clamp 

##Notes
Need the lsun data.
Follow the instruction in original implementation [link](https://github.com/martinarjovsky/WassersteinGAN)

##Sources
WGAN Model based on original paper [link](https://arxiv.org/abs/1701.07875)
Cross Attention is my own implementation


## To-Do
Rerun and recheck the results
