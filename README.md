# CT framework
Example call: 
```bash
python train.py --setup=3 --dataperc=75 --epochs=100 --lr=1e-5 --eps=1e-5 --alg=ADR --iterates=100 --valid=10 --batch-size=20 --detectors=128 --gpu=1 --noise=0.03 --load=False --seed=10 --wclip=False
```
## Description
This framework was created in order to help compare learned and variational approaches to CT reconstruction in a systematic way. The implementation is based on python libraries **odl** and **pyTorch**. The list of implemented algorithms include:

* FBP (Filtered back-projection) 
* TV (Total Variation)
* ADR (Adversarial Regularizer): https://arxiv.org/abs/1805.11572
* LG (Learned gradient descent): https://arxiv.org/abs/1704.04058
* LPD (Learned primal dual): https://arxiv.org/abs/1707.06474
* FL (Fully learned): https://nature.com/articles/nature25988.pdf
* FBP+U (FBP with a U-Net denoiser): https://arxiv.org/abs/1505.04597

In order to add your own algorithms to the list, create a new file in the **Algorithms** folder in the form *name*.py and use BaseAlg.py as the template.
