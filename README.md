# feature_level_bias_tors

> This is the implementation of our paper submitted to TORS (under review):
>
> Understanding and Counteracting Feature-Level Bias in Click-Through Rate Prediction
>
> By Jinqiu Jin, Sihao Ding, Wenjie Wang, Fuli Feng and Xiangnan He

1. To reproduce results in normal biased tests:

   ```shell
   cd ./code
   python reduction.py --dataset ml-1m --model nfm --alpha 0
   # dataset: {'ml-1m','book','kuairand'}
   # model: {'fm', 'nfm'}
   # alpha: 1->basemodel; 0->reduction
   ```

2. To reproduce results in debiased tests:

   ```shell
   cd ./code
   # model: {'fm', 'nfm'}
   # reconstruction: 0->basemodel; 1->reconstruction
   python reconstruction.py --reconstruction 1 --model nfm
   ```


Reference: [shenweichen/DeepCTR-Torch - GitHub](https://github.com/shenweichen/DeepCTR-Torch)
