# 基于LibFewShot的FD_Align论文复现

# 修改部分

## 增加Backbone

在 `backbone`的 `utils`文件夹里新增了 `bpe_simple_vocab_xxx`，`clip_module.py`，`clip_tokenizer.py`三个文件

在 `backbone`中新增了 `clip_vit.py`文件，即为核心backbone

在 `__init__.py`中将该backbone初始化，即为 `clip_vit32`

## 增加分类器

在 `finetuning`中添加了 `fd_align.py`文件，其中包含了训练，测试等一系列核心流程

在 `finetuning/fd_align_utils`中添加了 `class_name.py`，`openai_imagenet_temple.py`，分别是对应数据集顺序的类名排序，template列表，用于生成spj的分类器

## 增/改config文件

在 `config/headers/data.yaml`里面修改图片size为224，否则数据维度会有问题

在 `config/backbones`增加 `clip_vit.yaml`, 用于调用clip_vit32

在 `config`下增加了 `fd_align.yaml`，作为最终的配置文件（待修改）
