import torch

# 预定义的网络层torch.nn, 工具开发者已经帮助我们开发好的一些常用层,
# 比如，卷积层, lstm层, embedding层等, 不需要我们再重新造轮子.
import torch.nn as nn
import numpy as np
# 数学计算工具包
import math

# torch中变量封装函数Variable.
from torch.autograd import Variable

# 构建Embedding类实现文本嵌入层
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        """类的初始化函数, 有两个参数, d_model: 指词嵌入的维度, vocab: 指词表的大小."""
        # 接着就是使用super的方式指明继承nn.Module的初始化函数, 我们自己实现的所有层都会这样去写.
        super(Embeddings, self).__init__()
        # 之后就是调用nn中的预定义层Embedding, 获得一个词嵌入对象self.lut
        self.lut = nn.Embedding(vocab, d_model)
        # 最后就是将d_model传入类中
        self.d_model = d_model

    def forward(self, x):
        """可以将其理解为该层的前向传播逻辑，所有层中都会有此函数
           当传给该类的实例化对象参数时, 自动调用该类函数
           参数x: 因为Embedding层是首层, 所以代表输入给模型的文本通过词汇映射后的张量"""

        # 将x传给self.lut并与根号下self.d_model相乘作为结果返回
        return self.lut(x) * math.sqrt(self.d_model)

# 定义位置编码器类, 我们同样把它看做一个层, 因此会继承nn.Module
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        '''
        :param d_model: 代表词嵌入的维度
        :param dropout: 代表dropout层的置零比率
        :param max_len: 代表每隔句子的最大长度
        '''
        super(PositionalEncoding, self).__init__()

        # 实例化nn中预定义的Dropout层, 并将dropout传入其中, 获得对象self.dropout
        self.dropout = nn.Dropout(p=dropout)

        # 初始化一个位置编码矩阵, 它是一个0阵，矩阵的大小是max_len x d_model.
        pe = torch.zeros(max_len, d_model)

        # 初始化一个绝对位置矩阵, 在我们这里，词汇的绝对位置就是用它的索引去表示.
        # 所以我们首先使用arange方法获得一个连续自然数向量，然后再使用unsqueeze方法拓展向量维度使其成为矩阵，
        # 又因为参数传的是1，代表矩阵拓展的位置，会使向量变成一个max_len x 1 的矩阵，
        position = torch.arange(0, max_len).unsqueeze(1)

        # 定义一个变化矩阵div_tim，max_len*1
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        # 讲前面定义的变化矩阵进行奇数，偶数的分别赋值
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 这样我们就得到了位置编码矩阵pe, pe现在还只是一个二维矩阵，要想和embedding的输出（一个三维张量）相加，
        # 就必须拓展一个维度，所以这里使用unsqueeze拓展维度.
        pe = pe.unsqueeze(0)

        # 最后把pe位置编码矩阵注册成模型的buffer，什么是buffer呢，这个buffer不是模型中的参数
        # 我们把它认为是对模型效果有帮助的，但是却不是模型结构中超参数或者参数，不需要随着优化步骤进行更新的增益对象.
        # 注册之后我们就可以在模型保存后重加载时和模型结构与参数一同被加载.
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        :param x: 表示文本序列的词嵌入表示
        :return:
        '''
        # 在相加之前我们对pe做一些适配工作， 将这个三维张量的第二维也就是句子最大长度的那一维将切片到与输入的x的第二维相同即x.size(1)，
        # 因为我们默认max_len为5000一般来讲实在太大了，很难有一条句子包含5000个词汇，所以要进行与输入张量的适配.
        # 最后使用Variable进行封装，使其与x的样式相同，但是它是不需要进行梯度求解的，因此把requires_grad设置成false.
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        # 最后使用self.dropout对象进行'丢弃'操作, 并返回结果.
        return self.dropout(x)


# 构建掩码张量的函数
def subsequent_mask(size):
    '''
    :param size:掩码张量后两个维度，形成一个方阵
    :return:
    '''
    attn_shape = (1, size, size)

    # 然后使用np.ones方法向这个形状中添加1元素,形成上三角阵, 最后为了节约空间,
    # 再使其中的数据类型变为无符号8位整形unit8
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    # 最后将numpy类型转化为torch中的tensor, 内部做一个1 - 的操作,
    # 在这个其实是做了一个三角阵的反转, subsequent_mask中的每个元素都会被1减,
    # 如果是0, subsequent_mask中的该位置由0变成1
    # 如果是1, subsequent_mask中的该位置由1变成0
    return torch.from_numpy(1 - subsequent_mask)


def attention(query, key, value, mask=None, dropout=None):
    """
    注意力机制的实现
    :param query:注意力的输入
    :param key:注意力的输入
    :param value:注意力的输入
    :param mask:掩码张量
    :param dropout:传入的dropout实例化张量
    :return:
    """
    # 首先将query的最后一个维度取出来，代表词嵌入维度
    d_k = query.size(-1)

    # 按照注意力计算公式，将query和key的转置进行矩阵乘法，然后除以缩放稀疏
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 接着判断是否使用掩码张量
    if mask is not None:
        # 使用tensor的masked_fill方法, 将掩码张量和scores张量每个位置一一比较, 如果掩码张量处为0
        # 则对应的scores张量用-1e9这个值来替换, 如下演示
        scores = scores.masked_fill(mask == 0, -1e9)

    # 对scores的最后一维进行softmax操作, 使用F.softmax方法, 第一个参数是softmax对象, 第二个是目标维度.
    # 这样获得最终的注意力张量
    p_attn = F.softmax(scores, dim=-1)

    # 之后判断是否使用dropout进行随机置0
    if dropout is not None:
        # 将p_attn传入dropout对象中进行'丢弃'处理
        p_attn = dropout(p_attn)

    # 最后, 根据公式将p_attn与value张量相乘获得最终的query注意力表示, 同时返回注意力张量
    return torch.matmul(p_attn, value), p_attn


# 用于深度老被的copy工具包
import copy

# 首先需要定义克隆函数, 因为在多头注意力机制的实现中, 用到多个结构相同的线性层.
# 我们将使用clone函数将他们一同初始化在一个网络层列表对象中. 之后的结构中也会用到该函数
def clones(module, N):
    '''
    用于网络层的克隆函数
    :param module: 表示要克隆的网络
    :param N:代表需要克隆的数量
    :return:
    '''
    # 在函数中, 我们通过for循环对module进行N次深度拷贝, 使其每个module成为独立的层,
    # 然后将其放在nn.ModuleList类型的列表中存放.
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# 我们使用一个类来实现多头注意力机制的处理
class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        """在类的初始化时, 会传入三个参数，
        head代表头数，
        embedding_dim代表词嵌入的维度，
        dropout代表进行dropout操作时置0比率，默认是0.1.
        """
        super(MultiHeadedAttention, self).__init__()

        # 在函数中，首先使用了一个测试中常用的assert语句，判断h是否能被d_model整除，
        # 这是因为我们之后要给每个头分配等量的词特征.也就是embedding_dim/head个.
        assert embedding_dim % head == 0

        # 得到每个头获得的分割词向量维度d_k
        self.d_k = embedding_dim // head

        # 传入头数h
        self.head = head

        # 然后获得线性层对象，通过nn的Linear实例化，它的内部变换矩阵是embedding_dim x embedding_dim，然后使用clones函数克隆四个，
        # 为什么是四个呢，这是因为在多头注意力中，Q，K，V各需要一个，最后拼接的矩阵还需要一个，因此一共是四个.
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)

        # self.attn为None，它代表最后得到的注意力张量，现在还没有结果所以为None.
        self.attn = None

        # 最后就是一个self.dropout对象，它通过nn中的Dropout实例化而来，置0比率为传进来的参数dropout.
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """前向逻辑函数, 它的输入参数有四个，前三个就是注意力机制需要的Q, K, V，
           最后一个是注意力机制中可能需要的mask掩码张量，默认是None. """

        # 如果存在掩码张量mask
        if mask is not None:
            # 使用unsqueeze拓展维度
            mask = mask.unsqueeze(0)

        # 接着，我们获得一个batch_size的变量，他是query尺寸的第1个数字，代表有多少条样本.
        batch_size = query.size(0)

        # 之后就进入多头处理环节
        # 首先利用zip将输入QKV与三个线性层组到一起，然后使用for循环，将输入QKV分别传到线性层中，
        # 做完线性变换后，开始为每个头分割输入，这里使用view方法对线性变换的结果进行维度重塑，多加了一个维度h，代表头数，
        # 这样就意味着每个头可以获得一部分词特征组成的句子，其中的-1代表自适应维度，
        # 计算机会根据这种变换自动计算这里的值.然后对第二维和第三维进行转置操作，
        # 为了让代表句子长度维度和词向量维度能够相邻，这样注意力机制才能找到词义与句子位置的关系，
        # 从attention函数中可以看到，利用的是原始输入的倒数第一和第二维.这样我们就得到了每个头的输入.
        query, key, value = \
           [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
            for model, x in zip(self.linears, (query, key, value))]

        # 得到每个头的输入后，接下来就是将他们传入到attention中，
        # 这里直接调用我们之前实现的attention函数.同时也将mask和dropout传入其中.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 通过多头注意力计算后，我们就得到了每个头计算结果组成的4维张量，我们需要将其转换为输入的形状以方便后续的计算，
        # 因此这里开始进行第一步处理环节的逆操作，先对第二和第三维进行转置，然后使用contiguous方法，
        # 这个方法的作用就是能够让转置后的张量应用view方法，否则将无法直接使用，
        # 所以，下一步就是使用view重塑形状，变成和输入形状相同.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        # 最后使用线性层列表中的最后一个线性层对输入进行线性变换得到最终的多头注意力结构的输出.
        return self.linears[-1](x)


# 实现前馈全连接层
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        '''
        类的初始化
        :param d_model:代表词嵌入的维度,同时也是两个线性层的输入维度和输出维度
        :param d_ff:代表第一个线性层的输出维度,和第二个线性层的输入维度
        :param dropout:经过Dropout的置零处理
        '''
        super(PositionwiseFeedForward, self).__init__()

        # 定义两层全连接的线性层
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: 代表来自上一层的输出
        # 首先将x送入第一个线性层网络,然后进入relu，再经过dropout
        # 最后送入第二个线性层
        return self.w2(self.dropout(F.relu(self.w1(x))))



# 通过LayerNorm实现规范化层的类
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        '''
        :param features: 表示词嵌入的维度
        :param eps: 是一个足够小的数, 在规范化公式的分母中出现,防止分母为0.默认是1e-6
        '''
        super(LayerNorm, self).__init__()

        # 根据features的形状初始化两个参数张量a2，和b2，第一个初始化为1张量，
        # 也就是里面的元素都是1，第二个初始化为0张量，也就是里面的元素都是0，这两个张量就是规范化层的参数，
        # 因为直接对上一层得到的结果做规范化公式计算，将改变结果的正常表征，因此就需要有参数作为调节因子，
        # 使其即能满足规范化要求，又能不改变针对目标的表征.最后使用nn.parameter封装，代表他们是模型的参数。
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))

        # 把eps传到类中
        self.eps = eps

    def forward(self, x):
        """输入参数x代表来自上一层的输出"""
        # 在函数中，首先对输入变量x求其最后一个维度的均值，并保持输出维度与输入维度一致.
        # 接着再求最后一个维度的标准差，然后就是根据规范化公式，用x减去均值除以标准差获得规范化的结果，
        # 最后对结果乘以我们的缩放参数，即a2，*号代表同型点乘，即对应位置进行乘法操作，加上位移参数b2.返回即可.
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


# 使用SublayerConnection来实现子层连接结构的类
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        '''
        :param size:词嵌入维度的大小
        :param dropout:进行随机抑制的比率
        '''
        super(SublayerConnection, self).__init__()
        # 实例化了规范化对象self.norm
        self.norm = LayerNorm(size)
        # 又使用nn中预定义的droupout实例化一个self.dropout对象.
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        """前向逻辑函数中, 接收上一个层或者子层的输入作为第一个参数，
           将该子层连接中的子层函数作为第二个参数"""

        # 我们首先对输出进行规范化，然后将结果传给子层处理，之后再对子层进行dropout操作，
        # 随机停止一些网络中神经元的作用，来防止过拟合. 最后还有一个add操作，
        # 因为存在跳跃连接，所以是将输入x与dropout后的子层输出结果相加作为最终的子层连接输出.
        return x + self.dropout(sublayer(self.norm(x)))

# 使用EncoderLayer类实现编码器层
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        '''
        :param size: 词嵌入维度的大小
        :param self_attn:传入多头自注意力子层实例化对象,并且是自注意力机制
        :param feed_forward:前馈全连接层实例化对象
        :param dropout:置0比率
        '''
        super(EncoderLayer, self).__init__()
        # 把size传入其中
        self.size = size
        # 首先将self_attn和feed_forward传入其中.
        self.self_attn = self_attn
        self.feed_forward = feed_forward

        # 如图所示, 编码器层中有两个子层连接结构, 所以使用clones函数进行克隆
        self.sublayer = clones(SublayerConnection(size, dropout), 2)


    def forward(self, x, mask):
        """forward函数中有两个输入参数，x和mask，分别代表上一层的输出，和掩码张量mask."""
        # 里面就是按照结构图左侧的流程. 首先通过第一个子层连接结构，其中包含多头自注意力子层，
        # 然后通过第二个子层连接结构，其中包含前馈全连接子层. 最后返回结果.
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# 使用Encoder类来实现编码器
class Encoder(nn.Module):
    def __init__(self, layer, N):
        """初始化函数的两个参数分别代表编码器层和编码器层的个数"""
        super(Encoder, self).__init__()
        # 首先使用clones函数克隆N个编码器层放在self.layers中
        self.layers = clones(layer, N)
        # 再初始化一个规范化层, 它将用在编码器的最后面.
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """forward函数的输入和编码器层相同, x代表上一层的输出, mask代表掩码张量"""
        # 首先就是对我们克隆的编码器层进行循环，每次都会得到一个新的x，
        # 这个循环的过程，就相当于输出的x经过了N个编码器层的处理.
        # 最后再通过规范化层的对象self.norm进行处理，最后返回结果.
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# 使用DecoderLayer的类实现解码器层
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        '''
        :param size: 词嵌入的维度大小,同时也代表解码器层的尺寸
        :param self_attn: 多头自注意力对象,也就是说这个注意力机制需要Q=K=V
        :param src_attn: 多头注意力对象
        :param feed_forward: 前馈全连接层对象
        :param dropout: 置0比率
        '''

        super(DecoderLayer, self).__init__()
        # 在初始化函数中， 主要就是将这些输入传到类中
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # 按照结构图使用clones函数克隆三个子层连接对象.
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        '''
        :param x: 上一层的输入x
        :param memory: 编码器层的语义存储变量mermory
        :param source_mask: 源数据掩码张量
        :param target_mask: 目标数据掩码张
        :return:
        '''

        # 将memory表示成m方便之后使用
        m = memory

        # 将x传入第一个子层结构，第一个子层结构的输入分别是x和self-attn函数，因为是自注意力机制，所以Q,K,V都是x，
        # 最后一个参数是目标数据掩码张量，这时要对目标数据进行遮掩，因为此时模型可能还没有生成任何目标数据，
        # 比如在解码器准备生成第一个字符或词汇时，我们其实已经传入了第一个字符以便计算损失，
        # 但是我们不希望在生成第一个字符时模型能利用这个信息，因此我们会将其遮掩，同样生成第二个字符或词汇时，
        # 模型只能使用第一个字符或词汇信息，第二个字符以及之后的信息都不允许被模型使用.
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))

        # 接着进入第二个子层，这个子层中常规的注意力机制，q是输入x; k，v是编码层输出memory，
        # 同样也传入source_mask，但是进行源数据遮掩的原因并非是抑制信息泄漏，而是遮蔽掉对结果没有意义的字符而产生的注意力值，
        # 以此提升模型效果和训练速度. 这样就完成了第二个子层的处理.
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, source_mask))

        # 最后一个子层就是前馈全连接子层，经过它的处理后就可以返回结果.这就是我们的解码器层结构.
        return self.sublayer[2](x, self.feed_forward)


# 使用类Decoder来实现解码器
class Decoder(nn.Module):
    def __init__(self, layer, N):
        """初始化函数的参数有两个，第一个就是解码器层layer，第二个是解码器层的个数N."""
        super(Decoder, self).__init__()
        # 首先使用clones方法克隆了N个layer，然后实例化了一个规范化层.
        # 因为数据走过了所有的解码器层后最后要做规范化处理.
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        '''
        :param x: 目标数据的嵌入表示
        :param memory: 编码器层的输出
        :param source_mask: 源数据的掩码张量
        :param target_mask: 标数据的掩码张量
        :return:
        '''

        # 然后就是对每个层进行循环，当然这个循环就是变量x通过每一个层的处理，
        # 得出最后的结果，再进行一次规范化返回即可.
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)


# nn.functional工具包装载了网络层中那些只进行计算, 而没有参数的层
import torch.nn.functional as F

# 将线性层和softmax计算层一起实现, 因为二者的共同目标是生成最后的结构
# 因此把类的名字叫做Generator, 生成器类
class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        """初始化函数的输入参数有两个, d_model代表词嵌入维度, vocab_size代表词表大小."""
        super(Generator, self).__init__()
        # 首先就是使用nn中的预定义线性层进行实例化, 得到一个对象self.project等待使用,
        # 这个线性层的参数有两个, 就是初始化函数传进来的两个参数: d_model, vocab_size
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """前向逻辑函数中输入是上一层的输出张量x"""
        # 在函数中, 首先使用上一步得到的self.project对x进行线性变化,
        # 然后使用F中已经实现的log_softmax进行的softmax处理.
        # 在这里之所以使用log_softmax是因为和我们这个pytorch版本的损失函数实现有关, 在其他版本中将修复.
        # log_softmax就是对softmax的结果又取了对数, 因为对数函数是单调递增函数,
        # 因此对最终我们取最大的概率值没有影响. 最后返回结果即可.
        return F.log_softmax(self.project(x), dim=-1)


# 使用EncoderDecoder类来实现编码器-解码器结构
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        '''
        :param encoder: 编码器对象
        :param decoder: 解码器对象
        :param source_embed: 源数据嵌入函数
        :param target_embed: 目标数据嵌入函数
        :param generator: 输出部分的类别生成器对象
        '''

        super(EncoderDecoder, self).__init__()
        # 将参数传入到类中
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):
        '''
        :param source: 源数据
        :param target: 目标数据
        :param source_mask: 掩码张量
        :param target_mask: 掩码张量
        :return:
        '''

        # 在函数中, 将source, source_mask传入编码函数, 得到结果后,
        # 与source_mask，target，和target_mask一同传给解码函数.
        return self.decode(self.encode(source, source_mask), source_mask,
                            target, target_mask)

    def encode(self, source, source_mask):
        """编码函数, 以source和source_mask为参数"""
        # 使用src_embed对source做处理, 然后和source_mask一起传给self.encoder
        return self.encoder(self.src_embed(source), source_mask)

    def decode(self, memory, source_mask, target, target_mask):
        '''
        :param memory: 编码器的输出
        :param source_mask:源数据掩码张量
        :param target:目标数据
        :param target_mask:目标数据掩码张量
        :return:
        '''

        # 使用tgt_embed对target做处理, 然后和source_mask, target_mask, memory一起传给self.decoder
        return self.decoder(self.tgt_embed(target), memory, source_mask, target_mask)


def make_model(source_vocab, target_vocab, N=6,
               d_model=512, d_ff=2048, head=8, dropout=0.1):
    '''
    :param source_vocab: 源数据特征(词汇)总数
    :param target_vocab: 目标数据特征(词汇)总数
    :param N: 编码器和解码器堆叠数
    :param d_model: 词向量映射维度
    :param d_ff: 前馈全连接网络中变换矩阵的维度
    :param head: 多头注意力结构中的多头数
    :param dropout: 以及置零比率dropout
    :return:
    '''

    # 首先得到一个深度拷贝命令，接下来很多结构都需要进行深度拷贝，
    # 来保证他们彼此之间相互独立，不受干扰.
    c = copy.deepcopy

    # 实例化了多头注意力类，得到对象attn
    attn = MultiHeadedAttention(head, d_model)

    # 然后实例化前馈全连接类，得到对象ff
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    # 实例化位置编码类，得到对象position
    position = PositionalEncoding(d_model, dropout)

    # 根据结构图, 最外层是EncoderDecoder，在EncoderDecoder中，
    # 分别是编码器层，解码器层，源数据Embedding层和位置编码组成的有序结构，
    # 目标数据Embedding层和位置编码组成的有序结构，以及类别生成器层.
    # 在编码器层中有attention子层以及前馈全连接子层，
    # 在解码器层中有两个attention子层以及前馈全连接层.
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, source_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, target_vocab), c(position)),
        Generator(d_model, target_vocab))

    # 模型结构完成后，接下来就是初始化模型中的参数，比如线性层中的变换矩阵
    # 这里一但判断参数的维度大于1，则会将其初始化成一个服从均匀分布的矩阵，
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
