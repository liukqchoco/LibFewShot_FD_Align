from .fd_align_utils.class_name import mini_train
from .fd_align_utils.openai_imagenet_temple import openai_imagenet_template
from .finetuning_model import FinetuningModel
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple, List, Optional, Union, Dict
from ..backbone.clip_vit import get_ILF_kmeans_weights_classifier, ImageEncoder, load
from core.utils import accuracy

class CLIP_context(FinetuningModel):
    def __init__(
            self,
            metric: str = "cosine",
            scale_cls: float = 10.,
            normalize: bool = True,
            backbone_name: str = "resnet12",
            train_way: int = 5,
            val_way: int = 5,
            test_way: int = 5,
            train_shot: int = 5,
            val_shot: int = 5,
            test_shot: int = 5,
            num_query: int = 15,
            train_batch_size_per_gpu: int = 2,
            val_batch_size_per_gpu: int = 2,
            test_batch_size_per_gpu: int = 2,
            lr: float = 0.1,
            weight_decay: float = 5e-4,
            decay_scheduler: Optional[str] = "cosine",
            optim_type: str = "sgd",
            decay_epochs: Union[List, Tuple, None] = None,
            decay_power: Optional[float] = None,
            local_rank: int = -1,
            backbone_kwargs: Dict = {},
            cscale: float = 1.0,
            cnumber: int = 1,
            **kwargs
    ) -> None:
        """
        Args:
            metric: what metrics applied. "cosine" or "euclidean".
            scale_cls: The initial scale number which affects the
                    following softmax function.
            normalize: Whether normalize each spatial dimension of image features before average pooling.
            backbone_name: The name of the feature extractor,
                        which should match the correspond
                        file name in architectures.feature_extractor
            train_way: The number of classes within one training task.
            val_way: The number of classes within one training task.
            test_way: The number of classes within one training task.
            train_shot: The number of samples within each few-shot
                        support class during training.
                        For meta-learning only.
            val_shot: The number of samples within each few-shot
                    support class during validation.
            test_shot: The number of samples within each few-shot
                    support class during testing.
            num_query: The number of samples within each few-shot
                    query class.
            train_batch_size_per_gpu: The batch size of training per GPU.
            val_batch_size_per_gpu: The batch size of validation per GPU.
            test_batch_size_per_gpu: The batch size of testing per GPU.
            lr: The initial learning rate.
            weight_decay: The weight decay parameter.
            decay_scheduler: The scheduler of optimizer.
                            "cosine" or "specified_epochs".
            optim_type: The optimizer type.
                        "sgd" or "adam"
            decay_epochs: The list of decay epochs of decay_scheduler "specified_epochs".
            decay_power: The decay power of decay_scheduler "specified_epochs"
                        at eachspeicified epoch.
                        i.e., adjusted_lr = lr * decay_power
            backbone_kwargs: The parameters for creating backbone network.
        """
        super(CLIP_context, self).__init__(**kwargs)

        # FIXME:需要在这里记录属性
        self.train_way = train_way
        self.val_way = val_way
        self.test_way = test_way
        self.train_shot = train_shot
        self.val_shot = val_shot
        self.test_shot = test_shot

        self.num_query = num_query

        self.train_batch_size_per_gpu = train_batch_size_per_gpu
        self.val_batch_size_per_gpu = val_batch_size_per_gpu
        self.test_batch_size_per_gpu = test_batch_size_per_gpu

        self.train_label = torch.arange(self.train_way, dtype=torch.int8).repeat(num_query).type(torch.LongTensor).reshape(-1)
        self.val_label = torch.arange(self.val_way, dtype=torch.int8).repeat(num_query).type(torch.LongTensor).reshape(-1)
        self.test_label = torch.arange(self.test_way, dtype=torch.int8).repeat(num_query).type(torch.LongTensor).reshape(-1)

        # FIXME: 1. 视觉编码器
        # self.classifier = clip_head()
        self.classifier = PN_head(metric, scale_cls, normalize=normalize)
        clip_model = ImageEncoder(backbone_name)
        clip_model_, _, _ = load(backbone_name, jit=False)
        self.scale = cscale
        # self.context_scale = torch.nn.Parameter(torch.FloatTensor(1).fill_(1), requires_grad=True)

        self.zero_shot_clip = clip_model
        for param in self.zero_shot_clip.parameters():
            param.requires_grad = False
        # FIXME: 2. 文本编码器，里面的权重即为spj prototypes
        self.context_classifier = get_ILF_kmeans_weights_classifier(clip_model_, openai_imagenet_template, mini_train,
                                                                    cnumber, 60)
        for param in self.context_classifier.parameters():
            param.requires_grad = False
        del clip_model_
        self.loss_ctx = torch.nn.KLDivLoss()


    def set_forward(self, batch):
        way = self.test_way
        shot = self.test_shot
        test_batch_size_per_gpu = self.test_batch_size_per_gpu

        logits = self.inference_forward(batch, test_batch_size_per_gpu, way, shot)

        label = self.test_label
        label = torch.unsqueeze(label, 0).repeat(test_batch_size_per_gpu, 1).reshape(-1).to(logits.device)
        logits = logits.reshape(label.size(0), -1)

        acc = accuracy(logits, label)
        return logits, acc

    def inference_forward(self, batch, batch_size, way, shot):
        # PN head
        num_support_samples = way * shot
        data, _ = batch
        data = self.backbone(data)  # FIXME: visual embedding

        if len(data.shape) == 2:
            data = data.reshape([batch_size, -1] + list(data.shape[-1:]))
        else:
            data = data.reshape([batch_size, -1] + list(data.shape[-3:]))
        data_support = data[:, :num_support_samples]
        data_query = data[:, num_support_samples:]
        logits = self.classifier(data_query, data_support, way, shot)
        return logits

    def set_forward_loss(self, batch):
        way = self.train_way
        shot = self.train_shot
        train_batch_size_per_gpu = self.train_batch_size_per_gpu
        logits, ctx_loss = self.set_forward_adaptation(batch, train_batch_size_per_gpu, way, shot)

        label = self.train_label
        label = torch.unsqueeze(label, 0).repeat(self.train_batch_size_per_gpu, 1).reshape(-1).to(logits.device)
        logits = logits.reshape(label.size(0), -1)

        loss = F.cross_entropy(logits, label)
        acc = accuracy(logits, label)
        final_loss = loss + ctx_loss
        return logits, acc, final_loss

    def set_forward_adaptation(self, batch, batch_size, way, shot):
        num_support_samples = way * shot
        image, _ = batch
        data = self.backbone(image)
        with torch.no_grad():
            zero_data = self.zero_shot_clip(image)

        # context KL loss
        # use KL loss compute the context loss between the data and the zero shot data
        data = F.normalize(data, dim=1)
        zero_data = F.normalize(zero_data,
                                dim=1)
        ctx_loss = self.loss_ctx(torch.log(F.softmax(self.context_classifier(data), dim=1)),
                                 F.softmax(self.context_classifier(zero_data), dim=1))

        ctx_loss = self.scale * ctx_loss

        if len(data.shape) == 2:
            data = data.reshape([batch_size, -1] + list(data.shape[-1:]))
        else:
            data = data.reshape([batch_size, -1] + list(data.shape[-3:]))
        data_support = data[:, :num_support_samples]
        data_query = data[:, num_support_samples:]
        # classification logits
        logits = self.classifier(data_query, data_support, way, shot)

        return logits, ctx_loss

def get_model():
    return CLIP_context


class PN_head(nn.Module):
    r"""The metric-based protypical classifier from ``Prototypical Networks for Few-shot Learning''.

    Args:
        metric: Whether use cosine or enclidean distance.
        scale_cls: The initial scale number which affects the following softmax function.
        learn_scale: Whether make scale number learnable.
        normalize: Whether normalize each spatial dimension of image features before average pooling.
    """
    def __init__(
        self,
        metric: str = "cosine",
        scale_cls: int =10.0,
        learn_scale: bool = True,
        normalize: bool = True) -> None:
        super().__init__()
        assert metric in ["cosine", "enclidean"]
        if learn_scale:
            self.scale_cls = nn.Parameter(
                torch.FloatTensor(1).fill_(scale_cls), requires_grad=True
            )
        else:
            self.scale_cls = scale_cls
        self.metric = metric
        self.normalize = normalize

    def forward(self, features_test: Tensor, features_train: Tensor,
                way: int, shot: int) -> Tensor:
        r"""Take batches of few-shot training examples and testing examples as input,
            output the logits of each testing examples.

        Args:
            features_test: Testing examples. size: [batch_size, num_query, c, h, w]
            features_train: Training examples which has labels like:[abcdabcdabcd].
                            size: [batch_size, way*shot, c, h, w]
            way: The number of classes of each few-shot classification task.
            shot: The number of training images per class in each few-shot classification
                  task.
        Output:
            classification_scores: The calculated logits of testing examples.
                                   size: [batch_size, num_query, way]
        """
        if features_train.dim() == 5:
            if self.normalize:
                features_train=F.normalize(features_train, p=2, dim=2, eps=1e-12)
            features_train = F.adaptive_avg_pool2d(features_train, 1).squeeze_(-1).squeeze_(-1)
        assert features_train.dim() == 3

        batch_size = features_train.size(0)
        if self.metric == "cosine":
            features_train = F.normalize(features_train, p=2, dim=2, eps=1e-12)
        #prototypes: [batch_size, way, c]
        prototypes = torch.mean(features_train.reshape(batch_size, shot, way, -1),dim=1)
        prototypes = F.normalize(prototypes, p=2, dim=2, eps=1e-12)

        if self.normalize:
            features_test=F.normalize(features_test, p=2, dim=2, eps=1e-12)
        if features_test.dim() == 5:
            features_test = F.adaptive_avg_pool2d(features_test, 1).squeeze_(-1).squeeze_(-1)
        assert features_test.dim() == 3

        if self.metric == "cosine":
            features_test = F.normalize(features_test, p=2, dim=2, eps=1e-12)
            #[batch_size, num_query, c] * [batch_size, c, way] -> [batch_size, num_query, way]
            classification_scores = self.scale_cls * torch.bmm(features_test, prototypes.transpose(1, 2))

        elif self.metric == "euclidean":
            classification_scores = -self.scale_cls * L2SquareDist(features_test, prototypes)
        return classification_scores

def create_model(metric: str = "cosine",
        scale_cls: int =10.0,
        learn_scale: bool = True,
        normalize: bool = True):
    return PN_head(metric, scale_cls, learn_scale, normalize)


def L2SquareDist(A: Tensor, B: Tensor, average: bool = True) -> Tensor:
    r"""calculate parwise euclidean distance between two batchs of features.

    Args:
        A: Torch feature tensor. size:[Batch_size, Na, nC]
        B: Torch feature tensor. size:[Batch_size, Nb, nC]
    Output:
        dist: The calculated distance tensor. size:[Batch_size, Na, Nb]
    """
    assert A.dim() == 3
    assert B.dim() == 3
    assert A.size(0) == B.size(0) and A.size(2) == B.size(2)
    nB = A.size(0)
    Na = A.size(1)
    Nb = B.size(1)
    nC = A.size(2)

    # AB = A * B = [nB x Na x nC] * [nB x nC x Nb] = [nB x Na x Nb]
    AB = torch.bmm(A, B.transpose(1, 2))

    AA = (A * A).sum(dim=2, keepdim=True).view(nB, Na, 1)  # [nB x Na x 1]
    BB = (B * B).sum(dim=2, keepdim=True).view(nB, 1, Nb)  # [nB x 1 x Nb]
    # l2squaredist = A*A + B*B - 2 * A * B
    dist = AA.expand_as(AB) + BB.expand_as(AB) - 2 * AB
    if average:
        dist = dist / nC

    return dist