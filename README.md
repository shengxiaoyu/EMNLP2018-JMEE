# Jointly Multiple Events Extraction (JMEE)
论文 [EMNLP 2018 paper](https://arxiv.org/abs/1809.09078) 中 Jointly Multiple Events Extraction (JMEE) 事件抽取方法的复现，用于从中文离婚纠纷诉讼材料中抽取关注事件

### Requirement
- python 3
- [pytorch](http://pytorch.org) == 0.4.0
- [torchtext](https://github.com/pytorch/text) == 0.2.3
- [tensorboardX](https://github.com/lanpa/tensorboard-pytorch)
- [seqeval](https://github.com/chakki-works/seqeval)

To install the requirements, run `pip -r requirements.txt`.
这样安装pythorch会失败，可以将pytorch提出来单独安装，安装方式可以根据官网提供的命令行

### How to run the code?
After preprocessing the ACE 2005 dataset and put it under `ace-05-splits`, the main entrance is in `enet/run/ee/runner.py`.
We cannot include the data in this release due to licence issues.

But we offer a piece of data sample in `ace-05-splits/sample.json`, the format should be followed.
按'ace-05-splits/sample.json'格式准备好数据，需要用到stanfordnlp工具包，中文词向量是使用司法领域的数据训练的glove模型
