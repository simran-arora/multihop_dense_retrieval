This repository is for reproducing MDR results on the ConcurrentQA benchmark with and without the PAIR framework as in this paper: [Reasoning over Public and Private Data in Retrieval-Based Systems](https://arxiv.org/abs/2203.11027). The instructions to set up the environment, download data, and execute the training and evaluation scripts are detailed in the [ConcurrentQA Repository](https://github.com/facebookresearch/concurrentqa).


### Additional MDR Details

`MDR` is a simple and generalized dense retrieval method which recursively retrieves supporting text passages for answering complex open-domain questions. The repo provides code and pretrained retrieval models that produce **state-of-the-art** retrieval performance on two multi-hop QA datasets (the [HotpotQA](https://hotpotqa.github.io) dataset and the multi-hop subset of the [FEVER fact extraction and verification dataset](https://fever.ai)). 
See their ICLR paper for additional details: [Answering Complex Open-Domain Questions with Multi-Hop Dense Retrieval](https://arxiv.org/abs/2009.12756)

Please also cite the following if you use this code.
```
@article{xiong2020answering,
  title={Answering Complex Open-Domain Questions with Multi-Hop Dense Retrieval},
  author={Xiong, Wenhan and Li, Xiang Lorraine and Iyer, Srinivasan and Du, Jingfei and Lewis, Patrick and Wang, William Yang and Mehdad, Yashar and Yih, Wen-tau and Riedel, Sebastian and Kiela, Douwe and O{\u{g}}uz, Barlas},
  journal={International Conference on Learning Representations},
  year={2021}
}
```

### License
CC-BY-NC 4.0
