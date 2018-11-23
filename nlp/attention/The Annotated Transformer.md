

```python

```


```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")
%matplotlib inline
```

## 模型架构
大多数competitive neural sequence transduction models都有encoder-decoder架构([参考论文](https://arxiv.org/abs/1409.0473))。本文中，encoder将符号表示的输入序列$x_1, \dots, x_n$映射到一系列连续表示$Z=(z_1, \dots, z_n)$。给定一个z，decoder一次产生一个符号表示的序列输出$(y_1, \dots, y_m)$。对于每一步来说，模型都是自回归的([自回归介绍论文](https://arxiv.org/abs/1308.0850)),在生成下一个时消耗先前生成的所有符号作为附加输入。


```python
class EncoderDecoder(nn.Module):
    """
    A stanard Encoder-Decoder architecture.Base fro this and many other models.
    """
    
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        """ Take in and process masked src and target sequences. """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
```


```python
class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
```

Transformer这种结构，在encoder和decoder中使用堆叠的self-attention和point-wise全连接层。如下图的左边和右边所示：


```python
Image(filename='images/ModelNet-21.png')
```




![png](The%20Annotated%20Transformer_files/The%20Annotated%20Transformer_6_0.png)



## Encoder 和 Decoder Stacks
### Encoder
编码器由6个相同的layer堆叠而成


```python
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
```


```python
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```

这里在两个子层中都使用了残差连接([参考论文](https://arxiv.org/abs/1512.03385))，然后紧跟layer normalization([参考论文](https://arxiv.org/abs/1607.06450))


```python
class LayerNorm(nn.Module):
    """ Construct a layernorm model (See citation for details)"""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

也就是说，每个子层的输出是LayerNorm(x + Sublayer(x))，其中Sublayer(x)由子层实现。对于每一个子层，将其添加到子层输入并进行规范化之前，使用了Dropout([参考论文](http://jmlr.org/papers/v15/srivastava14a.html))

为了方便残差连接，模型中的所有子层和embedding层输出维度都是512


```python
class SublayerConnection(nn.Module):
    """ 
    A residual connection followed by a layer norm. Note for 
    code simplicity the norm is first as opposed to last .
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the sanme size. """
        return x + self.dropout(sublayer(self.norm(x)))
```

每层有两个子层。第一个子层是multi-head self-attention机制，第二层是一个简单的position-wise全连接前馈神经网络。


```python
class EncoderLayer(nn.Module):
    """Encoder is made up of self-attention and feed forward (defined below)"""
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
    
    def forward(self, x, mask):
        """Follow Figure 1 (left) for connection """
        x = self.sublayer[0](x, lambda x : self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
```

### Decoder
Decoder由6个相同layer堆成


```python
class Decoder(nn.Module):
    """Generic N layer decoder with masking """
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
```

每个encoder层除了两个子层外，还插入了第三个子层，即在encoder堆的输出上上执行multi-head注意力作用的层。类似于encoder，在每一个子层后面使用残差连接，并紧跟norm


```python
class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below)"""
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        """Follow Figure 1 (right) for connections"""
        m = memory
        x = self.sublayer[0](x, lambda x : self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x : self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
```

修改在decoder层堆中的self-atention 子层，防止位置关注后续位置。masking与使用一个position信息偏移的输出embedding相结合，确保对于position $i$ 的预测仅依赖于小于 $i$ 的position的输出


```python
def subsequent_mask(size):
    """Mask out subsequent positions. """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
```


```python
plt.figure(figsize=(5, 5))
plt.imshow(subsequent_mask(20)[0])
None
```


![png](The%20Annotated%20Transformer_files/The%20Annotated%20Transformer_22_0.png)


## Attention
注意力功能可以看做将一个query和一组key-value对映射到一个output，其中query、keys、values和output都是向量(vector)，输出是values的加权和，其中权重可以通过将query和对应的key输入到一个compatibility function来计算分配给每一个value的权重。

这里的attention其实可以叫做“Scaled Dot-Product Attention”。输入由$d_k$维度的queries和keys组成，values的维度是$d_v$。计算query和所有keys的点乘，然后除以$\sqrt{d_k}$，然后应用softmax函数来获取值的权重。$\sqrt{d_k}$起到调节作用，使得内积不至于太大（太大的话softmax后就非0即1了，不够“soft”了）。

实际计算中，一次计算一组queries的注意力函数，将其组成一个矩阵$Q$, 并且keys和values也分别组成矩阵$K$和$V$。此时，使用如下公式进行计算：
$$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$


```python
def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention ' """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) # matmul矩阵相乘
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
```

最常用的两种注意力实现机制包括： additive attention (cite), and dot-product (multiplicative) attention.
此处的实现是dot-product attention，不过多了$\sqrt{d_k}$。additive attention计算函数使用一个但隐藏层的前馈神经网络。
这两种实现机制在理论上复杂度是相似的，但是dot-product attention速度更快和更节省空间，因为可以使用高度优化的矩阵乘法来实现。

对于小规模values两种机制性能类差不多，但是对于大规模的values上，additive attention 性能优于 dot poduct。
原因分析：猜测可能是对于大规模values，内积会巨幅增长，将softmax函数推入有一个极小梯度的区域，造成性能下降（为了说明为什么内积变大，假设$q和k$ 是独立且平均值为0方差为1的随机变量，那么点乘$q*k = \sum^{d_k}_{i=1}q_ik_i$，其平均值为0，方差为1）为了抵消负面影响，使用$\sqrt{d_k}$来缩放内积


```python
from IPython.display import Image
Image("images/ModalNet-20.png")
```




![png](The%20Annotated%20Transformer_files/The%20Annotated%20Transformer_26_0.png)



Multi-head attention允许模型共同关注在不同位置的来自不同子空间的表示信息，只要一个单独的attention head，平均一下就会抑制上面所说的情况。此时，用公式表示如下：
$MultiHead(Q, K, V) = Concat(head-1, \dots, head_h)W^o$ 

其中$head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)$，$W_i^Q \in \mathcal{R}^{d_model * D_k}, W_i^K \in \mathcal{d_model * d_k}, W_i^V \in \mathcal{d_model*d_v} 并且 W_o \ in \mathcal{R}^{hd_v*d_{model}}$

此处，使用h=8平行的attention层或者heads，对每一层使用$d_k=d_v=d_{model}/h=64$


```python
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """ Take in model size and numbe of heads """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        """图片ModalNet-20的实现"""
        if mask is not None:
            # 同样的mask应用到所有heads
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1. 批量做linear投影 => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) 
                            for l, x in zip(self.linears, (query, key, value))]
        # 2. 批量应用attention机制在所有的投影向量上
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # 3. 使用view进行“Concat”并且进行最后一层的linear
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
```

### 模型中attention的应用
对于Transformer模型来说，使用三种不同的方式来使用multi-head attention
* 在“encoder-decoder attention”层，queries来自前一个decoder层，并且keys和values来自encoder的输出层。这使得在decoder层里的每一个位置信息对齐输入序列中的所有位置，这模拟了sequence-to-sequence模型中的典型的encoder-decoder注意力机制
* encoder层中包含了self-attention层。在self-attention层中，所有的keys、values和queries都来自同一处，在此例中，来自encoder的前一层。在encoder中的每一个position对齐encoder的前一层的所有position
* decoder层中的self-attention允许每一个position使用decoder的包括这个position在内的所有position。我们需要防止信息流左流来保证其自回归性。我们通过在缩放点乘 attention中使用mask技术（设置为负无穷）应用所有values，这个values是softmax层的输入，其对应非法连接（illegal connections）

## position-wise前馈神经网络
除了子层中的attention，在encoder和decoder的所有层中都包含一个全连接前馈神经网络，它将分别和共同应用于position。其中包括两个两个带有ReLU激活函数的线性变换：
$$FFN(x) = max(0, xW_1 + b_1)W_2 + b_2$$
对于不同的position，使用的线性变换虽然是相同的，但层与层之间的参数是不同的。这其实就是两个大小为1的一维卷积。输入和输出维度都是512，内层维度是2048


```python
class PositionwiseFeedForward(nn.Module):
    """ 
    FFN实现 
    d_model = 512
    d_ff = 2048
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
```

## Embeddings和Softmax
这里的Embeddings和传统的序列任务一样，使用训练好的Embedding，将输入token和输出token变成词向量，维度为d_model。
使用常见的线性变换和softmax函数将decoder的输出变为next-token的预测概率。
在我们的模型中，两个embedding层和pre-softmax层共享相同的权重矩阵。
在Embedding层，将$\sqrt{d_{model}}$乘以这些权重。


```python
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
```

## Positional Encoding
因为在此模型中不包含卷积和循环（层），这样的模型并不能捕捉序列的顺序！换句话说，如果将K,V按行打乱顺序（相当于句子中的词序打乱），那么Attention的结果还是一样的。对于时间序列来说，尤其是对于NLP中的任务来说，顺序是很重要的信息，它代表着局部甚至是全局的结构，学习不到顺序信息，那么效果将会大打折扣（比如机器翻译中，有可能只把每个词都翻译出来了，但是不能组织成合理的句子）。为了使模型能够充分利用序列信息，必须为模型注入tokens的相对或绝对信息。为此，加入“position embedding”到encoder和decoder层底部的输入层中。
position embedding和embeddings有相同的维度，都是d_model。所以这两个emebdding可以求和。
对于position embedding，有很多选择，可以参考这篇论文。[点击查看](https://arxiv.org/pdf/1705.03122.pdf)
在此模型中，使用不同频率的正弦和余弦函数：
$$PE_{(pos, 2i)}(p） = sin(pos/10000^{2i/d_{model}})，PE_{(pos, 2i+1)}(p)= cos(pos/10000^{2i/d_{modle}})$$
pos代表position，i代表维度。所以，position编码的每个维度对应正弦曲线。这里的意思是将id为p的位置映射为一个dpos维的位置向量，这个向量的第i个元素的数值就是$PE_i(p)$。波长是从$2\pi到1000*2\pi$的几何级数。
之所以选择这个函数，由于我们有sin(α+β)=sinαcosβ+cosαsinβ以及cos(α+β)=cosαcosβ−sinαsinβ，这表明位置p+k的向量可以表示成位置p的向量的线性变换，这提供了表达相对位置信息的可能性。
对于任何偏移量k，$PE_{pos+k}$ 可以表示为$PE_{pos}$的线性函数。

除此之外，在encoder和decoder中的positional编码中和embeddings求和中都使用了dropout。对于base model，$P_{drop} = 0.1$


```python
class PositionalEncoding(nn.Module):
    """PE函数实现"""
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-(math.log(10000.0)/ d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
    
```


```python
plt.figure(figsize=(15, 5))
pe = PositionalEncoding(20, 0)
y = pe.forward(Variable(torch.zeros(1, 100, 20)))
plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
plt.legend(["dim %d" %p for p in [4, 5, 6, 7]])
None
```


![png](The%20Annotated%20Transformer_files/The%20Annotated%20Transformer_36_0.png)


论文里比较过直接训练出来的位置向量和上述公式计算出来的位置向量，效果是接近的。因此显然我们更乐意使用公式构造的Position Embedding了，因为允许模型扩展到比训练时候序列更长的序列长度。


```python
def make_model(src_vacab, tgt_vocab, N=6, d_model=512, d_ff =2048, h=8, dropout=0.1):
    """ 构建模型"""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vacab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )
    
    # !!!import for the work
    # 使用Glorot/ fan_avg初始化参数
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
```


```python
tmp_model = make_model(10, 10, 2)
```


```python
tmp_model
```




    EncoderDecoder(
      (encoder): Encoder(
        (layers): ModuleList(
          (0): EncoderLayer(
            (self_attn): MultiHeadedAttention(
              (linears): ModuleList(
                (0): Linear(in_features=512, out_features=512, bias=True)
                (1): Linear(in_features=512, out_features=512, bias=True)
                (2): Linear(in_features=512, out_features=512, bias=True)
                (3): Linear(in_features=512, out_features=512, bias=True)
              )
              (dropout): Dropout(p=0.1)
            )
            (feed_forward): PositionwiseFeedForward(
              (w_1): Linear(in_features=512, out_features=2048, bias=True)
              (w_2): Linear(in_features=2048, out_features=512, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (sublayer): ModuleList(
              (0): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
              (1): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
            )
          )
          (1): EncoderLayer(
            (self_attn): MultiHeadedAttention(
              (linears): ModuleList(
                (0): Linear(in_features=512, out_features=512, bias=True)
                (1): Linear(in_features=512, out_features=512, bias=True)
                (2): Linear(in_features=512, out_features=512, bias=True)
                (3): Linear(in_features=512, out_features=512, bias=True)
              )
              (dropout): Dropout(p=0.1)
            )
            (feed_forward): PositionwiseFeedForward(
              (w_1): Linear(in_features=512, out_features=2048, bias=True)
              (w_2): Linear(in_features=2048, out_features=512, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (sublayer): ModuleList(
              (0): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
              (1): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
            )
          )
        )
        (norm): LayerNorm(
        )
      )
      (decoder): Decoder(
        (layers): ModuleList(
          (0): DecoderLayer(
            (self_attn): MultiHeadedAttention(
              (linears): ModuleList(
                (0): Linear(in_features=512, out_features=512, bias=True)
                (1): Linear(in_features=512, out_features=512, bias=True)
                (2): Linear(in_features=512, out_features=512, bias=True)
                (3): Linear(in_features=512, out_features=512, bias=True)
              )
              (dropout): Dropout(p=0.1)
            )
            (src_attn): MultiHeadedAttention(
              (linears): ModuleList(
                (0): Linear(in_features=512, out_features=512, bias=True)
                (1): Linear(in_features=512, out_features=512, bias=True)
                (2): Linear(in_features=512, out_features=512, bias=True)
                (3): Linear(in_features=512, out_features=512, bias=True)
              )
              (dropout): Dropout(p=0.1)
            )
            (feed_forward): PositionwiseFeedForward(
              (w_1): Linear(in_features=512, out_features=2048, bias=True)
              (w_2): Linear(in_features=2048, out_features=512, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (sublayer): ModuleList(
              (0): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
              (1): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
              (2): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
            )
          )
          (1): DecoderLayer(
            (self_attn): MultiHeadedAttention(
              (linears): ModuleList(
                (0): Linear(in_features=512, out_features=512, bias=True)
                (1): Linear(in_features=512, out_features=512, bias=True)
                (2): Linear(in_features=512, out_features=512, bias=True)
                (3): Linear(in_features=512, out_features=512, bias=True)
              )
              (dropout): Dropout(p=0.1)
            )
            (src_attn): MultiHeadedAttention(
              (linears): ModuleList(
                (0): Linear(in_features=512, out_features=512, bias=True)
                (1): Linear(in_features=512, out_features=512, bias=True)
                (2): Linear(in_features=512, out_features=512, bias=True)
                (3): Linear(in_features=512, out_features=512, bias=True)
              )
              (dropout): Dropout(p=0.1)
            )
            (feed_forward): PositionwiseFeedForward(
              (w_1): Linear(in_features=512, out_features=2048, bias=True)
              (w_2): Linear(in_features=2048, out_features=512, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (sublayer): ModuleList(
              (0): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
              (1): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
              (2): SublayerConnection(
                (norm): LayerNorm(
                )
                (dropout): Dropout(p=0.1)
              )
            )
          )
        )
        (norm): LayerNorm(
        )
      )
      (src_embed): Sequential(
        (0): Embeddings(
          (lut): Embedding(10, 512)
        )
        (1): PositionalEncoding(
          (dropout): Dropout(p=0.1)
        )
      )
      (tgt_embed): Sequential(
        (0): Embeddings(
          (lut): Embedding(10, 512)
        )
        (1): PositionalEncoding(
          (dropout): Dropout(p=0.1)
        )
      )
      (generator): Generator(
        (proj): Linear(in_features=512, out_features=10, bias=True)
      )
    )



## 模型训练
首先定义一个包含源句子和目标句子的批处理对象，同事构建masks

### Batches and Masking


```python
class Batch:
    """ 在训练期间使用mask处理数据 """
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        """ 创造一个mask来屏蔽补全词和字典外的词进行屏蔽"""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
```

### Training and Loss compute


```python
def run_epoch(data_iter, model, loss_compute):
    """ 标准训练和日志函数 """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss : %f Tokens per Sec: %f " % (i, loss/ batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens
```

### Training Data and Batching
论文中使用的数据集较大，这里使用的是torchtext函数


```python
global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    """ 保持数据批量增加，并计算tokens+padding的总数 """
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)
```

### Optimizer
使用Adam优化器， 其中$\beta_1 = 0.9, \beta_2 = 0.98, \epsilon = 10^{-1}$

使用如下方式调整学习率：
$$lrate = d_{model}^{-0.5}\cdot \min(step\_num^{-0.5}, step\_num \cdot warmup\_step^{-1.5})$$
先随着训练step线性增加，之后将其与步数的倒数平方根成比例地减小，论文中warmupsteps = 4000


```python
# 这个部分很重要，需要这样设置模型参数
```


```python
class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        """ 更新参数和学习率 """
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        """ lrate 实现"""
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))
    
def get_std_up(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000, 
                   torch.optim.Adam(model.param_groups(), 
                                    lr = 0, betas = (0.9, 0.98), eps = 1e-9))
```

#### 针对不同模型大小和优化超参数的此模型的曲线示例。


```python
opts = [NoamOpt(512, 1, 4000, None),
       NoamOpt(512, 1, 8000, None),
       NoamOpt(256, 1, 4000, None)]

plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
plt.legend(["512:4000", "512:8000", "256:4000"])
None
```


![png](The%20Annotated%20Transformer_files/The%20Annotated%20Transformer_52_0.png)


### Regularization
#### Label Smoothing 标签平滑
论文中训练的时候使用$\epsilon = 0.1$进行标签平滑（[参考论文](https://arxiv.org/abs/1512.00567)），这样会增加复杂性，因为模型学得更加不确定性，但提高了准确性和BLEU分数。

在实际实现时，这里使用KL div loss实现标签平滑。没有使用one-hot目标分布，而是创建了一个分布，对于整个词汇分布表，这个分布含有正确单词度和剩余部分平滑块的置信度


```python
class LabelSmoothing(nn.Module):
    """ 标签平滑实现 """
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
            
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))            
```

在这里，我们可以看到基于置信度如何将语料分布到单词的示例。


```python
# 标签平滑的例子
crit = LabelSmoothing(5, 0, 0.4)
predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                            [0, 0.2, 0.7, 0.1, 0],
                            [0, 0.2, 0.7, 0.1, 0]])

v = crit(Variable(predict.log()), Variable(torch.LongTensor([2, 1, 0])))
# 展示目标label的期望分布
plt.imshow(crit.true_dist)
None
```


![png](The%20Annotated%20Transformer_files/The%20Annotated%20Transformer_56_0.png)


#### 如果对于一个给定选择非常有信息，标签平滑实际上开始惩罚模型



```python
crit = LabelSmoothing(5, 0, 0.1)
def loss(x):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x/d, 1/d, 1/d, 1/d],])
    return crit(Variable(predict.log()),
               Variable(torch.LongTensor([1]))).data[0]
plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
None
```


![png](The%20Annotated%20Transformer_files/The%20Annotated%20Transformer_58_0.png)


### 实战：第一个例子
给定一个来自小词汇表的随机输入符号集，目标是生成相同的符号

### 合成数据


```python
def data_gen(V, batch, nbatches):
    """ 生成一个随机数据用于 src-tgt复制任务"""
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad = False)
        tgt = Variable(data, requires_grad = False)
        yield Batch(src, tgt, 0)
```

### 损失计算


```python
class SimpleLossCompute:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)        
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                             y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data[0] * norm
```

### 贪心Decoding


```python
V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V,V, N=2)
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400, 
                    torch.optim.Adam(model.parameters(), lr=0, betas= (0.9, 0.98), eps = 1e-9))

for epoch in range(10):
    model.train()
    run_epoch(data_gen(V, 30, 20), model, SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    print(run_epoch(data_gen(V, 30, 5), model, SimpleLossCompute(model.generator, criterion, None)))
```

    Epoch Step: 1 Loss : 3.328415 Tokens per Sec: 576.238429 
    Epoch Step: 1 Loss : 1.918249 Tokens per Sec: 1231.307105 
    1.9602290630340575
    Epoch Step: 1 Loss : 1.989585 Tokens per Sec: 628.817043 
    Epoch Step: 1 Loss : 1.696718 Tokens per Sec: 950.082767 
    1.7239755392074585
    Epoch Step: 1 Loss : 1.785850 Tokens per Sec: 634.098792 
    Epoch Step: 1 Loss : 1.525769 Tokens per Sec: 1203.840811 
    1.5165136814117433
    Epoch Step: 1 Loss : 1.778025 Tokens per Sec: 653.846467 
    Epoch Step: 1 Loss : 1.222946 Tokens per Sec: 1136.071379 
    1.1622309684753418
    Epoch Step: 1 Loss : 1.238613 Tokens per Sec: 668.932793 
    Epoch Step: 1 Loss : 0.997262 Tokens per Sec: 1121.050903 
    1.0589515805244445
    Epoch Step: 1 Loss : 1.218332 Tokens per Sec: 655.329661 
    Epoch Step: 1 Loss : 0.739608 Tokens per Sec: 1086.835112 
    0.7851933479309082
    Epoch Step: 1 Loss : 0.782321 Tokens per Sec: 616.218030 
    Epoch Step: 1 Loss : 0.397338 Tokens per Sec: 1076.720783 
    0.4492629885673523
    Epoch Step: 1 Loss : 0.600206 Tokens per Sec: 641.399357 
    Epoch Step: 1 Loss : 0.224609 Tokens per Sec: 1124.049134 
    0.299098813533783
    Epoch Step: 1 Loss : 0.451857 Tokens per Sec: 631.001445 
    Epoch Step: 1 Loss : 0.361520 Tokens per Sec: 916.635636 
    0.3899755597114563
    Epoch Step: 1 Loss : 0.986942 Tokens per Sec: 664.370915 
    Epoch Step: 1 Loss : 0.413342 Tokens per Sec: 1099.376349 
    0.3377967059612274


为了简单起见，使用贪婪的解码方式来预测对应翻译


```python
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask,
                          Variable(ys),
                          Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys
model.eval()
src = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
src_mask = Variable(torch.ones(1, 1, 10))
print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))
```

    
        1     2     3     4     5     8     6     7     9    10
    [torch.LongTensor of size 1x10]
    


## 真实的例子
下面使用the IWSLT German-English Translation task做为真实任务。这个任务要比原始论文中的VMT任务小的多，但是它也能反应整个系统的优势。这里也介绍了如何使用多GPU编程使得它速度更快

### Data Loading
这里使用torchtext加载数据和spacy进行token切分


```python
from torchtext import data, datasets

if True:
    import spacy
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')
    
    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]
    
    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]
    
    BOS_WORD = '<s>'
    EOS_WORD = '</S>'
    BLANK_WORD = "<blank>"
    SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD, eos_token=EOS_WORD, pad_token=BLANK_WORD)
    
    MAX_LEN = 100
    train, val, test = datasets.IWSLT.splits(
        exts=('.de', '.en'), fields=(SRC, TGT),
        filter_pred=lambda x : len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN)
    MIN_FREQ = 2
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)
    
```

    
    [93m    Warning: no model found for 'de'[0m
    
        Only loading the 'de' tokenizer.
    
    
    [93m    Warning: no model found for 'en'[0m
    
        Only loading the 'en' tokenizer.
    


分批次对于训练速度来说非常重要。我们希望拥有非常均匀分割的批次，同事拥有最小的填充（padding）。为了做到这一点，需要修改torchtext的默认的批处理方式。下面的代码修改其默认的批处理方式，以确保我们能够搜索足够多的句子以找到好的批处理。

### Iterators


```python
class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
            
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)

```

### Multi-GPU Training
为了加速训练，使用多GPU。实现方法不是transformer独有的，所以细节上不多做介绍。实现方法是在训练时将单词分块分配到不通的GPU上，以便并行处理。我们使用pytorch的并行处理原语来实现。

- replicate - 将模块拆分到不同的GPU上 
- scatter - 将批次拆分到不同的GPU上
- parallel_apply -  将模块应用于位于不同gpus上的批次
- gather - 将分散的数据拉回到一个gpu上。
- nn.DataParallel - 一个特殊的模块包装器，在评估之前调用。


```python
class MultiGPULossCompute:
    "A multi-gpu loss compute and train function."
    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, 
                                               devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size
        
    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator, 
                                                devices=self.devices)
        out_scatter = nn.parallel.scatter(out, 
                                          target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets, 
                                      target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[Variable(o[:, i:i+chunk_size].data, 
                                    requires_grad=self.opt is not None)] 
                           for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss. 
            y = [(g.contiguous().view(-1, g.size(-1)), 
                  t[:, i:i+chunk_size].contiguous().view(-1)) 
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l = nn.parallel.gather(loss, 
                                   target_device=self.devices[0])
            l = l.sum()[0] / normalize
            total += l.data[0]

            # Backprop loss to output of transformer
            if self.opt is not None:
                l.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # Backprop all loss through transformer.            
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad, 
                                    target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return total * normalize
```


```python
# 需要使用的GPU
devices = [0] # 如果只有一个GPU，使用devices=[0]
if True:
    pad_idx = TGT.vocab.stoi["<blank>"]
    model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
    model.cuda()
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda()
    BATCH_SIZE = 12000
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0, repeat=False, 
                            sort_key=lambda x : (len(x.src), len(x.trg)),
                           batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0, repeat=False,
                           sort_key=lambda x : (len(x.src), len(x.trg)),
                           batch_size_fn=batch_size_fn, train=False)
    model_par = nn.DataParallel(model, device_ids=devices)
None
# 这里需要很大的内存，报内存错误很正常，可以直接用下面训练好的
# 或者调小BATCH_SIZE
```

### 训练模型


```python
!wget https://s3.amazonaws.com/opennmt-models/iwslt.pt
```

    --2018-11-23 10:41:12--  https://s3.amazonaws.com/opennmt-models/iwslt.pt
    Resolving s3.amazonaws.com (s3.amazonaws.com)... 52.216.99.165
    Connecting to s3.amazonaws.com (s3.amazonaws.com)|52.216.99.165|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 467317581 (446M) [application/x-www-form-urlencoded]
    Saving to: ‘iwslt.pt’
    
    iwslt.pt            100%[===================>] 445.67M  4.19MB/s    in 2m 6s   
    
    2018-11-23 10:43:20 (3.52 MB/s) - ‘iwslt.pt’ saved [467317581/467317581]
    



```python
if True:
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                       torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(10):
        model_par.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter),
                  model_par, 
                  MultiGPULossCompute(model.generator, criterion, devices=devices, opt=model_opt))
        model_par.eval()
        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), 
                         model_par, 
                         MultiGPULossCompute(model.generator, criterion, devices=devices, opt=None))
        print("loss is: %f" % loss)
else:
    model = torch.load("iwslt.pt")
    
```

    Epoch Step: 1 Loss : 9.115798 Tokens per Sec: 27.246552 
    Epoch Step: 51 Loss : 8.795138 Tokens per Sec: 112.815511 
    Epoch Step: 101 Loss : 8.123654 Tokens per Sec: 111.014568 
    Epoch Step: 151 Loss : 6.664777 Tokens per Sec: 116.761415 
    Epoch Step: 201 Loss : 6.139312 Tokens per Sec: 121.009845 
    Epoch Step: 251 Loss : 6.167897 Tokens per Sec: 108.962893 
    Epoch Step: 301 Loss : 6.259430 Tokens per Sec: 132.455119 
    Epoch Step: 351 Loss : 6.787484 Tokens per Sec: 112.652976 
    Epoch Step: 401 Loss : 5.871879 Tokens per Sec: 116.497260 
    Epoch Step: 451 Loss : 5.735006 Tokens per Sec: 119.585556 
    Epoch Step: 501 Loss : 5.359651 Tokens per Sec: 124.397652 
    Epoch Step: 551 Loss : 5.068015 Tokens per Sec: 123.167267 
    Epoch Step: 601 Loss : 4.455752 Tokens per Sec: 125.204276 
    Epoch Step: 651 Loss : 5.691089 Tokens per Sec: 123.898633 
    Epoch Step: 701 Loss : 4.096532 Tokens per Sec: 116.935452 
    Epoch Step: 751 Loss : 5.427895 Tokens per Sec: 113.504319 
    Epoch Step: 801 Loss : 4.077328 Tokens per Sec: 114.052573 
    Epoch Step: 851 Loss : 5.073078 Tokens per Sec: 124.525615 
    Epoch Step: 901 Loss : 6.252522 Tokens per Sec: 111.752486 
    Epoch Step: 951 Loss : 5.530307 Tokens per Sec: 122.574079 
    Epoch Step: 1001 Loss : 7.020243 Tokens per Sec: 112.812934 
    Epoch Step: 1051 Loss : 6.083723 Tokens per Sec: 108.350083 
    Epoch Step: 1101 Loss : 5.993296 Tokens per Sec: 125.369539 
    Epoch Step: 1151 Loss : 5.358730 Tokens per Sec: 113.058275 
    Epoch Step: 1201 Loss : 5.063932 Tokens per Sec: 123.037357 
    Epoch Step: 1251 Loss : 4.229568 Tokens per Sec: 130.335474 
    Epoch Step: 1301 Loss : 4.840786 Tokens per Sec: 125.532243 
    Epoch Step: 1351 Loss : 4.929336 Tokens per Sec: 122.771668 
    Epoch Step: 1401 Loss : 4.490815 Tokens per Sec: 117.422497 
    Epoch Step: 1451 Loss : 5.564615 Tokens per Sec: 113.588785 
    Epoch Step: 1501 Loss : 6.767705 Tokens per Sec: 113.181958 
    Epoch Step: 1551 Loss : 5.262781 Tokens per Sec: 109.249956 
    Epoch Step: 1601 Loss : 5.217732 Tokens per Sec: 124.954539 
    Epoch Step: 1651 Loss : 6.329094 Tokens per Sec: 110.064008 
    Epoch Step: 1701 Loss : 4.028631 Tokens per Sec: 121.401211 
    Epoch Step: 1751 Loss : 5.967369 Tokens per Sec: 110.828476 
    Epoch Step: 1801 Loss : 5.388975 Tokens per Sec: 123.864972 
    Epoch Step: 1851 Loss : 6.298844 Tokens per Sec: 126.828853 
    Epoch Step: 1901 Loss : 5.405447 Tokens per Sec: 126.701156 
    Epoch Step: 1951 Loss : 6.067888 Tokens per Sec: 107.605533 
    Epoch Step: 2001 Loss : 6.092531 Tokens per Sec: 112.409601 
    Epoch Step: 2051 Loss : 5.238168 Tokens per Sec: 124.090227 
    Epoch Step: 2101 Loss : 6.180753 Tokens per Sec: 124.890180 
    Epoch Step: 2151 Loss : 5.203738 Tokens per Sec: 116.304379 
    Epoch Step: 2201 Loss : 5.281256 Tokens per Sec: 114.753139 
    Epoch Step: 2251 Loss : 5.884039 Tokens per Sec: 132.970013 
    Epoch Step: 2301 Loss : 5.384135 Tokens per Sec: 116.574694 
    Epoch Step: 2351 Loss : 4.522669 Tokens per Sec: 116.200856 
    Epoch Step: 2401 Loss : 5.505316 Tokens per Sec: 136.415777 
    Epoch Step: 2451 Loss : 6.195974 Tokens per Sec: 105.207516 
    Epoch Step: 2501 Loss : 5.338635 Tokens per Sec: 111.450436 
    Epoch Step: 2551 Loss : 6.587592 Tokens per Sec: 120.631876 
    Epoch Step: 2601 Loss : 5.582253 Tokens per Sec: 108.563041 
    Epoch Step: 2651 Loss : 6.719680 Tokens per Sec: 124.482789 
    Epoch Step: 2701 Loss : 6.663082 Tokens per Sec: 111.772424 
    Epoch Step: 2751 Loss : 6.241520 Tokens per Sec: 106.365521 
    Epoch Step: 2801 Loss : 6.269626 Tokens per Sec: 120.436012 
    Epoch Step: 2851 Loss : 5.431211 Tokens per Sec: 126.338207 
    Epoch Step: 2901 Loss : 4.765350 Tokens per Sec: 142.394894 
    Epoch Step: 2951 Loss : 5.357254 Tokens per Sec: 118.471949 
    Epoch Step: 3001 Loss : 5.476350 Tokens per Sec: 120.072949 
    Epoch Step: 3051 Loss : 4.086349 Tokens per Sec: 118.580939 
    Epoch Step: 3101 Loss : 5.351029 Tokens per Sec: 100.739349 
    Epoch Step: 3151 Loss : 4.154559 Tokens per Sec: 111.941574 
    Epoch Step: 3201 Loss : 6.364028 Tokens per Sec: 114.184378 
    Epoch Step: 3251 Loss : 5.718968 Tokens per Sec: 99.218677 
    Epoch Step: 3301 Loss : 5.274074 Tokens per Sec: 117.763638 
    Epoch Step: 3351 Loss : 5.384217 Tokens per Sec: 125.402623 
    Epoch Step: 3401 Loss : 5.846756 Tokens per Sec: 132.068638 
    Epoch Step: 3451 Loss : 5.346969 Tokens per Sec: 115.760696 
    Epoch Step: 3501 Loss : 4.732840 Tokens per Sec: 128.605812 
    Epoch Step: 3551 Loss : 5.474891 Tokens per Sec: 118.020504 
    Epoch Step: 3601 Loss : 4.852159 Tokens per Sec: 134.320444 
    Epoch Step: 3651 Loss : 5.451114 Tokens per Sec: 131.592711 
    Epoch Step: 3701 Loss : 5.287390 Tokens per Sec: 121.867178 
    Epoch Step: 3751 Loss : 4.070665 Tokens per Sec: 125.037012 
    Epoch Step: 3801 Loss : 4.605243 Tokens per Sec: 116.971368 
    Epoch Step: 3851 Loss : 5.196218 Tokens per Sec: 118.281831 
    Epoch Step: 3901 Loss : 5.077266 Tokens per Sec: 121.393991 
    Epoch Step: 3951 Loss : 3.693632 Tokens per Sec: 130.745685 
    Epoch Step: 4001 Loss : 6.163212 Tokens per Sec: 115.424976 
    Epoch Step: 4051 Loss : 4.347610 Tokens per Sec: 127.553092 
    Epoch Step: 4101 Loss : 5.545203 Tokens per Sec: 127.837643 
    Epoch Step: 4151 Loss : 3.849491 Tokens per Sec: 110.226187 
    Epoch Step: 4201 Loss : 3.980309 Tokens per Sec: 114.186194 
    Epoch Step: 4251 Loss : 5.881281 Tokens per Sec: 101.449358 
    Epoch Step: 4301 Loss : 5.177263 Tokens per Sec: 120.402445 
    Epoch Step: 4351 Loss : 5.984916 Tokens per Sec: 123.156741 
    Epoch Step: 4401 Loss : 5.315350 Tokens per Sec: 127.725499 
    Epoch Step: 4451 Loss : 5.140157 Tokens per Sec: 120.348082 
    Epoch Step: 4501 Loss : 4.589716 Tokens per Sec: 115.936711 
    Epoch Step: 4551 Loss : 5.617508 Tokens per Sec: 123.089776 
    Epoch Step: 4601 Loss : 5.993092 Tokens per Sec: 105.903955 
    Epoch Step: 4651 Loss : 4.702520 Tokens per Sec: 130.786161 
    Epoch Step: 4701 Loss : 3.969887 Tokens per Sec: 108.160083 
    Epoch Step: 4751 Loss : 4.690508 Tokens per Sec: 113.869501 
    Epoch Step: 4801 Loss : 6.810272 Tokens per Sec: 113.617623 
    Epoch Step: 4851 Loss : 6.173089 Tokens per Sec: 126.269595 
    Epoch Step: 4901 Loss : 6.111300 Tokens per Sec: 125.321265 
    Epoch Step: 4951 Loss : 4.817343 Tokens per Sec: 119.167878 
    Epoch Step: 5001 Loss : 4.348187 Tokens per Sec: 118.838030 
    Epoch Step: 5051 Loss : 5.152521 Tokens per Sec: 137.800332 
    Epoch Step: 5101 Loss : 4.129012 Tokens per Sec: 93.108974 
    Epoch Step: 5151 Loss : 3.953586 Tokens per Sec: 109.937495 
    Epoch Step: 5201 Loss : 5.148945 Tokens per Sec: 121.888139 
    Epoch Step: 5251 Loss : 4.364405 Tokens per Sec: 109.109888 
    Epoch Step: 5301 Loss : 5.165133 Tokens per Sec: 114.984078 
    Epoch Step: 5351 Loss : 3.795167 Tokens per Sec: 113.023181 
    Epoch Step: 5401 Loss : 6.682058 Tokens per Sec: 118.737053 
    Epoch Step: 5451 Loss : 4.287959 Tokens per Sec: 104.789962 
    Epoch Step: 5501 Loss : 4.816484 Tokens per Sec: 115.577422 
    Epoch Step: 5551 Loss : 5.741656 Tokens per Sec: 127.177478 
    Epoch Step: 5601 Loss : 5.204136 Tokens per Sec: 116.299978 
    Epoch Step: 5651 Loss : 4.231252 Tokens per Sec: 104.789153 
    Epoch Step: 5701 Loss : 4.711532 Tokens per Sec: 115.029154 
    Epoch Step: 5751 Loss : 5.010115 Tokens per Sec: 96.724771 
    Epoch Step: 5801 Loss : 5.679179 Tokens per Sec: 103.678304 
    Epoch Step: 5851 Loss : 4.840335 Tokens per Sec: 130.473356 
    Epoch Step: 5901 Loss : 6.077025 Tokens per Sec: 115.137809 
    Epoch Step: 5951 Loss : 4.814110 Tokens per Sec: 123.587690 
    Epoch Step: 6001 Loss : 5.069083 Tokens per Sec: 114.077725 
    Epoch Step: 6051 Loss : 4.493855 Tokens per Sec: 109.636051 
    Epoch Step: 6101 Loss : 4.939024 Tokens per Sec: 105.390777 
    Epoch Step: 6151 Loss : 5.475710 Tokens per Sec: 108.860868 
    Epoch Step: 6201 Loss : 4.954838 Tokens per Sec: 114.393112 
    Epoch Step: 6251 Loss : 4.933300 Tokens per Sec: 133.104268 
    Epoch Step: 6301 Loss : 3.424345 Tokens per Sec: 123.297081 
    Epoch Step: 6351 Loss : 4.743899 Tokens per Sec: 123.668512 
    Epoch Step: 6401 Loss : 5.173761 Tokens per Sec: 116.578303 
    Epoch Step: 6451 Loss : 5.513897 Tokens per Sec: 115.043922 
    Epoch Step: 6501 Loss : 4.724627 Tokens per Sec: 124.206271 
    Epoch Step: 6551 Loss : 6.129314 Tokens per Sec: 101.478472 
    Epoch Step: 6601 Loss : 5.877912 Tokens per Sec: 114.402926 
    Epoch Step: 6651 Loss : 5.689779 Tokens per Sec: 114.263450 
    Epoch Step: 6701 Loss : 4.473933 Tokens per Sec: 137.414472 
    Epoch Step: 6751 Loss : 4.558855 Tokens per Sec: 119.674591 
    Epoch Step: 6801 Loss : 5.443909 Tokens per Sec: 117.412783 
    Epoch Step: 6851 Loss : 5.363852 Tokens per Sec: 132.389490 
    Epoch Step: 6901 Loss : 5.363576 Tokens per Sec: 116.070900 
    Epoch Step: 6951 Loss : 4.576881 Tokens per Sec: 119.039487 
    Epoch Step: 7001 Loss : 5.497922 Tokens per Sec: 113.802638 
    Epoch Step: 7051 Loss : 6.701012 Tokens per Sec: 118.306036 
    Epoch Step: 7101 Loss : 2.813069 Tokens per Sec: 112.697843 
    Epoch Step: 7151 Loss : 4.721593 Tokens per Sec: 118.721749 
    Epoch Step: 7201 Loss : 4.454715 Tokens per Sec: 126.265449 
    Epoch Step: 7251 Loss : 5.204863 Tokens per Sec: 110.260285 
    Epoch Step: 7301 Loss : 6.093715 Tokens per Sec: 113.003313 
    Epoch Step: 7351 Loss : 3.692578 Tokens per Sec: 114.283425 
    Epoch Step: 7401 Loss : 4.344957 Tokens per Sec: 106.621936 
    Epoch Step: 7451 Loss : 5.794671 Tokens per Sec: 115.961877 
    Epoch Step: 7501 Loss : 4.730721 Tokens per Sec: 137.014758 
    Epoch Step: 7551 Loss : 5.345826 Tokens per Sec: 103.827868 
    Epoch Step: 7601 Loss : 4.509674 Tokens per Sec: 118.272483 
    Epoch Step: 7651 Loss : 4.735533 Tokens per Sec: 124.188938 
    Epoch Step: 7701 Loss : 4.103126 Tokens per Sec: 118.460493 
    Epoch Step: 7751 Loss : 5.854985 Tokens per Sec: 129.768230 
    Epoch Step: 7801 Loss : 6.111448 Tokens per Sec: 119.035855 
    Epoch Step: 7851 Loss : 5.163021 Tokens per Sec: 112.247344 
    Epoch Step: 7901 Loss : 4.288763 Tokens per Sec: 131.180708 
    Epoch Step: 7951 Loss : 5.100498 Tokens per Sec: 117.128035 
    Epoch Step: 8001 Loss : 5.268844 Tokens per Sec: 125.332613 
    Epoch Step: 8051 Loss : 4.714831 Tokens per Sec: 129.184088 
    Epoch Step: 8101 Loss : 4.888225 Tokens per Sec: 123.157112 
    Epoch Step: 8151 Loss : 3.944123 Tokens per Sec: 127.816168 
    Epoch Step: 8201 Loss : 4.901572 Tokens per Sec: 115.956655 
    Epoch Step: 8251 Loss : 4.748166 Tokens per Sec: 108.353690 
    Epoch Step: 8301 Loss : 4.617119 Tokens per Sec: 112.925223 
    Epoch Step: 8351 Loss : 5.873656 Tokens per Sec: 116.868465 
    Epoch Step: 8401 Loss : 5.075790 Tokens per Sec: 109.263429 
    Epoch Step: 8451 Loss : 4.742764 Tokens per Sec: 118.775604 
    Epoch Step: 8501 Loss : 5.432733 Tokens per Sec: 118.623984 
    Epoch Step: 8551 Loss : 5.449299 Tokens per Sec: 100.890450 
    Epoch Step: 8601 Loss : 4.598395 Tokens per Sec: 109.322259 
    Epoch Step: 8651 Loss : 4.820777 Tokens per Sec: 121.645203 
    Epoch Step: 8701 Loss : 5.348703 Tokens per Sec: 120.845176 
    Epoch Step: 8751 Loss : 4.150349 Tokens per Sec: 116.405742 
    Epoch Step: 8801 Loss : 5.885915 Tokens per Sec: 126.198338 
    Epoch Step: 8851 Loss : 5.417064 Tokens per Sec: 109.893480 
    Epoch Step: 8901 Loss : 4.649410 Tokens per Sec: 113.371114 
    Epoch Step: 8951 Loss : 5.403794 Tokens per Sec: 131.876288 
    Epoch Step: 9001 Loss : 4.591927 Tokens per Sec: 109.430485 
    Epoch Step: 9051 Loss : 4.999243 Tokens per Sec: 113.865406 
    Epoch Step: 9101 Loss : 4.565268 Tokens per Sec: 114.044977 
    Epoch Step: 9151 Loss : 4.651077 Tokens per Sec: 99.303425 
    Epoch Step: 9201 Loss : 5.085640 Tokens per Sec: 112.334779 
    Epoch Step: 9251 Loss : 4.164530 Tokens per Sec: 110.798513 
    Epoch Step: 9301 Loss : 4.909528 Tokens per Sec: 113.906289 
    Epoch Step: 9351 Loss : 5.305653 Tokens per Sec: 118.921177 
    Epoch Step: 9401 Loss : 5.303745 Tokens per Sec: 105.413858 
    Epoch Step: 9451 Loss : 5.203192 Tokens per Sec: 102.531900 



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-52-27ea9f2d920a> in <module>()
          6         run_epoch((rebatch(pad_idx, b) for b in train_iter),
          7                   model_par,
    ----> 8                   MultiGPULossCompute(model.generator, criterion, devices=devices, opt=model_opt))
          9         model_par.eval()
         10         loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), 


    <ipython-input-32-1a41203ea234> in run_epoch(data_iter, model, loss_compute)
          5     total_loss = 0
          6     tokens = 0
    ----> 7     for i, batch in enumerate(data_iter):
          8         out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
          9         loss = loss_compute(out, batch.trg_y, batch.ntokens)


    <ipython-input-52-27ea9f2d920a> in <genexpr>(.0)
          4     for epoch in range(10):
          5         model_par.train()
    ----> 6         run_epoch((rebatch(pad_idx, b) for b in train_iter),
          7                   model_par,
          8                   MultiGPULossCompute(model.generator, criterion, devices=devices, opt=model_opt))


    ~/.conda/envs/pytorch3/lib/python3.6/site-packages/torchtext/data/iterator.py in __iter__(self)
        149                         minibatch.sort(key=self.sort_key, reverse=True)
        150                 yield Batch(minibatch, self.dataset, self.device,
    --> 151                             self.train)
        152             if not self.repeat:
        153                 return


    ~/.conda/envs/pytorch3/lib/python3.6/site-packages/torchtext/data/batch.py in __init__(self, data, dataset, device, train)
         25                 if field is not None:
         26                     batch = [getattr(x, name) for x in data]
    ---> 27                     setattr(self, name, field.process(batch, device=device, train=train))
         28 
         29     @classmethod


    ~/.conda/envs/pytorch3/lib/python3.6/site-packages/torchtext/data/field.py in process(self, batch, device, train)
        186         """
        187         padded = self.pad(batch)
    --> 188         tensor = self.numericalize(padded, device=device, train=train)
        189         return tensor
        190 


    ~/.conda/envs/pytorch3/lib/python3.6/site-packages/torchtext/data/field.py in numericalize(self, arr, device, train)
        315                 arr = arr.contiguous()
        316         else:
    --> 317             arr = arr.cuda(device)
        318             if self.include_lengths:
        319                 lengths = lengths.cuda(device)


    ~/.conda/envs/pytorch3/lib/python3.6/site-packages/torch/_utils.py in _cuda(self, device, async)
         67         else:
         68             new_type = getattr(torch.cuda, self.__class__.__name__)
    ---> 69             return new_type(self.size()).copy_(self, async)
         70 
         71 


    KeyboardInterrupt: 


经过训练，我们可以对模型进行解码从而进行翻译。这里我们简单的对验证集的第一个句子进行翻译。这个数据集非常小，所以使用贪婪搜索准确度非常高。


```python
for i, batch in enumerate(valid_iter):
    src = batch.src.transpose(0, 1)[:1]
    src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
    out = greedy_decode(model, src, src_mask, max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
    print("Translation: ", end="\t")
    for i in range(1, out.size(1)):
        sym = TGT.vocab.itos[out[0, i]]
        if sym == "</s>":
            break
        print(sym, end=" ")
    print()    
    print("Target:", end="\t")
    for i in range(1, batch.trg.size(0)):
        sym = TGT.vocab.itos[batch.trg.data[i, 0]]
        if sym == '</s>':
            break
        print(sym, end=" ")
    print()
    break
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-46-80b2c4ba57eb> in <module>()
    ----> 1 for i, batch in enumerate(valid_iter):
          2     src = batch.src.transpose(0, 1)[:1]
          3     src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
          4     out = greedy_decode(model, src, src_mask, max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
          5     print("Translation: ", end="\t")


    NameError: name 'valid_iter' is not defined


## 其他部分：BPE、搜索、平均
到这里已经汉高了transformer模型的所有部分。有四个部分没有详细介绍。附加的特性在[OpenNMT-py](https://github.com/opennmt/opennmt-py)有介绍

1) BPE/ Word-piece: 我们可以使用库来首先将数据预处理为子字单元。参考Rico Sennrich’s 的[subword-nmt](https://github.com/rsennrich/subword-nmt)实现。这些模型将数据转换为如下格式：

▁Die ▁Protokoll datei ▁kann ▁ heimlich ▁per ▁E - Mail ▁oder ▁FTP ▁an ▁einen ▁bestimmte n ▁Empfänger ▁gesendet ▁werden .

2）共享Embeddings(Shared Embeddings): 使用BPE会共享vocabulary，我们也可以共享source / target / generator之间的权重向量。具体可以参考[这篇文献](https://arxiv.org/abs/1608.05859)。使用可以像下面这样：


```python
if False:
    model.src_embed[0].lut.weight = model.tgt_embeddings[0].lut.weight
    model.generator.lut.weight = model.tgt_embed[0].lut.weight
```

3) Beam Search: 这个有点复杂。可以参考[OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/translate/Beam.py)的pytorch实现
4）Model Averaging: 论文里平均最后K个检查点（模型结果）以达到一个集成效果，如果我们有一堆模型，我们可以按照如下方式去做：


```python
def average(model, models):
    """平均模型，创建一个新模型"""
    for ps in zip(*[m.params() for m in [model] + models])
    p[0].copy_(torch.sum(*ps[1:]) / len(ps[1:]))
```


      File "<ipython-input-53-6a7063a2cb31>", line 3
        for ps in zip(*[m.params() for m in [model] + models])
                                                              ^
    SyntaxError: invalid syntax



### 结果
论文里使用8个P100 GPUS训练了3.5天。在WMT 2014 English-to-French translation task中达到最好效果。dropout使用的是0.1.


```python
Image(filename="images/result.png")
```




![png](The%20Annotated%20Transformer_files/The%20Annotated%20Transformer_86_0.png)



这里的代码是基础版本，完整的系统版本可以参考 ([Example Models](http://opennmt.net/Models-py/)).
可以使用下面训练好的模型


```python
# !wget https://s3.amazonaws.com/opennmt-models/en-de-model.pt
```


```python
model, SRC, TGT = torch.load("en-de-model.pt")
```

    /home/tenyun/.conda/envs/pytorch3/lib/python3.6/site-packages/torch/serialization.py:325: SourceChangeWarning: source code of class 'torch.nn.modules.linear.Linear' has changed. you can retrieve the original source code by accessing the object's source attribute or set `torch.nn.Module.dump_patches = True` and use the patch tool to revert the changes.
      warnings.warn(msg, SourceChangeWarning)
    /home/tenyun/.conda/envs/pytorch3/lib/python3.6/site-packages/torch/serialization.py:325: SourceChangeWarning: source code of class 'torch.nn.modules.sparse.Embedding' has changed. you can retrieve the original source code by accessing the object's source attribute or set `torch.nn.Module.dump_patches = True` and use the patch tool to revert the changes.
      warnings.warn(msg, SourceChangeWarning)



```python
model.eval()
sent = "▁The ▁log ▁file ▁can ▁be ▁sent ▁secret ly ▁with ▁email ▁or ▁FTP ▁to ▁a ▁specified ▁receiver".split()
src = torch.LongTensor([[SRC.stoi[w] for w in sent]])
src = Variable(src)
src_mask = (src != SRC.stoi["<blank>"]).unsqueeze(-2)
out = greedy_decode(model, src, src_mask, 
                    max_len=60, start_symbol=TGT.stoi["<s>"])
print("Translation:", end="\t")
trans = "<s> "
for i in range(1, out.size(1)):
    sym = TGT.itos[out[0, i]]
    if sym == "</s>": break
    trans += sym + " "
print(trans)
```

    Translation:	<s> ▁Die ▁Protokoll datei ▁kann ▁ heimlich ▁per ▁E - Mail ▁oder ▁FTP ▁an ▁einen ▁bestimmte n ▁Empfänger ▁gesendet ▁werden . 


### Attention可视化
我们可以进一步看看，看看每一层注意力层发生了什么


```python
tgt_sent = trans.split()
def draw(data, x, y, ax):
    seaborn.heatmap(data, xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0, cbar=False, ax=ax)
    
for layer in range(1, 6, 2):
    fig, axs = plt.subplots(1, 4, figsize=(20, 10))
    print("Encoder Layer", layer+1)
    for h in range(4):
        draw(model.encoder.layers[layer].self_attn.attn[0, h].data,
            sent, sent if h==0 else [], ax=axs[h])
    plt.show()
    
for layer in range(1, 6, 2):
    fig, axs = plt.subplots(1, 4, figsize=(20, 10))
    print("Decoder Self Layer", layer + 1)
    for h in range(4):
        draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent),:len(tgt_sent)],
            tgt_sent, tgt_sent if h == 0 else [] , ax = axs[h])
    plt.show()
    print("Decoder Src Layer", layer+1)
    fig, axs = plt.subplots(1, 4, figsize=(20, 10))
    for h in range(4):
        draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(sent)],
            sent, tgt_sent if h==0 else [], ax=axs[h])
    plt.show()
```

    Encoder Layer 2



![png](The%20Annotated%20Transformer_files/The%20Annotated%20Transformer_92_1.png)


    Encoder Layer 4



![png](The%20Annotated%20Transformer_files/The%20Annotated%20Transformer_92_3.png)


    Encoder Layer 6



![png](The%20Annotated%20Transformer_files/The%20Annotated%20Transformer_92_5.png)


    Decoder Self Layer 2



![png](The%20Annotated%20Transformer_files/The%20Annotated%20Transformer_92_7.png)


    Decoder Src Layer 2



![png](The%20Annotated%20Transformer_files/The%20Annotated%20Transformer_92_9.png)


    Decoder Self Layer 4



![png](The%20Annotated%20Transformer_files/The%20Annotated%20Transformer_92_11.png)


    Decoder Src Layer 4



![png](The%20Annotated%20Transformer_files/The%20Annotated%20Transformer_92_13.png)


    Decoder Self Layer 6



![png](The%20Annotated%20Transformer_files/The%20Annotated%20Transformer_92_15.png)


    Decoder Src Layer 6



![png](The%20Annotated%20Transformer_files/The%20Annotated%20Transformer_92_17.png)



```python

```


```python

```
