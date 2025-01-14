# 基于LibFewShot的FD_Align论文复现

# 修改部分

## 增加Backbone

在`backbone`的`utils`文件夹里新增了`bpe_simple_vocab_xxx`，`clip_module.py`，`clip_tokenizer.py`三个文件

在`backbone`中新增了`clip_vit.py`文件，即为核心backbone

在`__init__.py`中将该backbone初始化，即为`clip_vit32`

## 增加分类器

在`finetuning`中添加了`fd_align.py`文件，其中包含了训练，测试等一系列核心流程
