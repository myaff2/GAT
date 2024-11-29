from typing import Optional, Tuple, Union

import math
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn import LayerNorm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value


class ResGATv2Conv(MessagePassing):
    r"""The GATv2 operator from the `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ paper, which fixes the
    static attention problem of the standard
    :class:`~torch_geometric.conv.GATConv` layer.
    Since the linear layers in the standard GAT are applied right after each
    other, the ranking of attended nodes is unconditioned on the query node.
    In contrast, in :class:`GATv2`, every node can attend to any other node.

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_k]
        \right)\right)}.

    If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
    the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_j \, \Vert \, \mathbf{e}_{i,j}]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_k \, \Vert \, \mathbf{e}_{i,k}]
        \right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        fill_value (float or torch.Tensor or str, optional): The way to
            generate edge features of self-loops
            (in case :obj:`edge_dim != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        share_weights (bool, optional): If set to :obj:`True`, the same matrix
            will be applied to the source and the target node of every edge.
            (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})` or
          :math:`((|\mathcal{V}_t|, H * F_{out})` if bipartite.
          If :obj:`return_attention_weights=True`, then
          :math:`((|\mathcal{V}|, H * F_{out}),
          ((2, |\mathcal{E}|), (|\mathcal{E}|, H)))`
          or :math:`((|\mathcal{V_t}|, H * F_{out}), ((2, |\mathcal{E}|),
          (|\mathcal{E}|, H)))` if bipartite
    """
    # 可选的Tensor，通常用于存储计算出的注意力系数。
    _alpha: OptTensor

    def __init__(
        self,
        # 输入特征的维度。如果是一个整数，则表示输入的所有节点特征的维度相同；
        # 如果是一个元组（Tuple），则第一个值是源节点特征的维度，第二个值是目标节点特征的维度。这在使用图注意力网络时特别有用，因为源节点和目标节点可能具有不同的特征维度。
        in_channels: Union[int, Tuple[int, int]],
        # 输出特征的维度。这个值决定了每个头部（head）输出的特征数。
        out_channels: int,
        heads: int = 1,
        # 决定是否将多个头的输出拼接在一起（concatenate）。如果为True，则将所有头的输出拼接成一个向量；如果为False，则对所有头的输出取平均值。
        concat: bool = True,
        # Leaky ReLU函数中的负斜率（Negative Slope）。Leaky ReLU 是一种激活函数，允许少量的负值通过，以帮助避免“死亡ReLU”问题。
        negative_slope: float = 0.2,
        # 在训练期间对注意力权重应用的丢弃率（Dropout rate），用于防止过拟合。它随机地将一些注意力权重设置为零。
        dropout: float = 0.0,
        # 决定是否为每个节点添加自环（Self-loops）。自环是指一个节点连向自身的边，通常在图神经网络中用于丰富节点的自表示信息。
        add_self_loops: bool = True,
        # 边特征的维度。如果图中的边也有特征，这个参数定义了这些特征的维度。如果为None，则表示没有边特征。
        edge_dim: Optional[int] = None,
        # 当添加自环时，指定如何填充这些边的特征值。可以是一个浮点数、Tensor或者一个字符串（例如'mean'，表示用边特征的均值填充）。
        fill_value: Union[float, Tensor, str] = 'mean',
        # 决定是否在计算输出时添加偏置（bias）项。偏置项可以帮助模型学习到偏移量，改善模型的拟合能力。
        bias: bool = True,
        # 决定是否在源节点和目标节点之间共享权重。如果为True，则源节点和目标节点使用相同的权重矩阵进行线性变换。
        share_weights: bool = False,
        # 决定是否添加残差连接（Residual Connection）。残差连接是一种架构，通过在输出中添加输入，可以缓解深层网络中的梯度消失问题。
        residual: bool = False,
        **kwargs,
    ):
        # 调用父类的构造函数进行初始化，node_dim=0 指定了节点的维度，**kwargs 用于传递其他可选参数。
        super().__init__(node_dim=0, **kwargs)
        # 这是一个断言，确保 out_channels（输出通道数）不为0。如果为0，程序将抛出错误。
        assert out_channels != 0

        # 这些是对传入参数的赋值，将它们存储为类的属性。
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.share_weights = share_weights

        # 根据 in_channels 是否为整数来选择不同的处理方式。
        # 如果 in_channels 是整数，则线性层 lin_l 和 lin_r 都将被初始化为从 in_channels 映射到 heads * out_channels 的线性变换。
        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=bias,
                                weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels, heads * out_channels,
                                    bias=bias, weight_initializer='glorot')
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels,
                                bias=bias, weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels[1], heads * out_channels,
                                    bias=bias, weight_initializer='glorot')

        # 是一个注意力参数，是可学习的，用于计算图中的注意力权重。
        self.att = Parameter(torch.Tensor(1, heads, out_channels))
        
        # 如果 edge_dim 被定义，那么就会为边特征添加一个线性变换层 lin_edge。
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
        else:
            self.lin_edge = None

        # 如果设置了 residual 和 concat，则初始化残差连接和层归一化（LayerNorm）。否则，将不使用残差连接。
        if residual and concat:
            self.residual = Linear(in_channels, heads * out_channels, bias=bias, weight_initializer='glorot')
            self.layerNorm = LayerNorm(heads * out_channels)
        elif residual and not concat:
            self.residual = Linear(in_channels, out_channels, bias=bias, weight_initializer='glorot')
            self.layerNorm = LayerNorm(out_channels)
        else:
            self.register_parameter('residual', None)
            self.register_parameter('layerNorm', None)
            
        # 根据 bias 和 concat 的设置，初始化偏置（bias）参数。
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        # 调用 reset_parameters 方法，重置模型的所有参数
        self.reset_parameters()

    # 这个方法用于重置模型的参数，将线性层的参数初始化为 glorot（均匀分布），并将偏置参数初始化为0。
    def reset_parameters(self):
        super().reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    # 这是前向传播函数。
    # x 可以是单个Tensor或一对Tensor（PairTensor），edge_index 是图的边索引，edge_attr 是边的特征，return_attention_weights 决定是否返回注意力权重。
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None,
                return_attention_weights: bool = None):
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""Runs the forward pass of the module.

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        # 将头数和输出通道数分别赋给 H 和 C，便于后续使用。
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        # 判断 x 的类型，如果是Tensor，进行不同的处理。如果是PairTensor，则分别处理 x_l 和 x_r。
        # 假设输入 x 的形状为 [300, 1030]
        # print("x_begin:",x.size())
        if isinstance(x, Tensor):
            assert x.dim() == 2
            # x_l 的形状会变成 [300, 8, 64]
            x_l = self.lin_l(x).view(-1, H, C)
            # print("x_l:",x_l.size())
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
        
        # print("x_l_begin:",x_l.size())
        # print("x_r_begin:",x_r.size())

        assert x_l is not None
        assert x_r is not None
        
        
        # 如果设置了 add_self_loops，则为图添加自环边。这里区分了 edge_index 是普通Tensor和稀疏张量的情况。
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        # 调用 propagate 函数，这是图神经网络中的关键步骤，用于执行消息传递（Message Passing）。
        # 在 PyTorch Geometric 中，self.propagate 是 MessagePassing 类提供的方法，负责实现图神经网络中的消息传递机制。propagate 方法会自动调用以下三个方法中的一个或多个：
        #     message: 计算消息内容，即从源节点传递到目标节点的信息。
        #     aggregate: 聚合收到的消息，通常通过求和或平均。
        #     update: 更新节点的特征，使用聚合后的消息进行节点特征的更新。

        # self.propagate 在实际运行时会根据你传递的 edge_index 和 x 等信息自动调用 message 方法。
        # propagate 函数在处理图的边时会将节点特征扩展到与边的数量匹配，从而执行消息传递和聚合操作。
        # 即[300, 8, 64]->[13378, 8, 64]
        # print("x_l:",x_l.size())
        # print("x_r:",x_r.size())
        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr,
                             size=None)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        # 如果设置了 concat，则将所有头的输出拼接起来，否则取平均值。
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        # 如果设置了残差连接，则将残差添加到输出中。
        if self.residual is not None:
            out = out + self.layerNorm(self.residual(x).view(-1, self.heads * self.out_channels))

        # 如果设置了偏置参数，则将偏置添加到输出中。
        if self.bias is not None:
            out = out + self.bias

        # 根据 return_attention_weights 的设置，决定是否返回注意力权重。
        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    # 这个函数用于在消息传递过程中计算每个边的消息，包含节点和边特征的组合以及注意力权重的计算。
    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # 在 message 方法中，x_i 和 x_j 分别代表了目标节点和源节点的特征。它们是图神经网络中消息传递机制的关键部分，具体来说：
        #     x_j: 代表源节点的特征。这些是消息传递的“发送者”，即消息将从这些节点出发，传递给与之相连的目标节点。
        #     x_i: 代表目标节点的特征。这些是消息传递的“接收者”，即消息会传递到这些节点上。

        # 在图神经网络中，消息传递的过程可以理解为：每个节点通过边将自身的特征传递给它的邻居节点，然后邻居节点对接收到的特征进行聚合，更新自身的特征。在这个过程里：
        #     x_j 是从源节点发出的特征，表示邻居节点的信息。
        #     x_i 是目标节点本身的特征，表示正在接收信息的节点。
        # print("x_i:",x_i.size())
        # print("x_j:",x_j.size())
        x = x_i + x_j
        # Dimension of x: torch.Size([6839, 8, 64]).包含 6839 个节点，每个节点有 8 个注意力头，每个头产生 64 维的特征。
        # print("Dimension of x:", x.size())

        if edge_attr is not None:
            # 如果 edge_attr 是一维的（即每条边只有一个特征值），它会将其形状调整为二维，使其变成形状为 (-1, 1) 的张量。这是为了确保边特征在后续操作中能够正确处理。
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            # 这层线性变换的作用是将边的特征投射到与节点特征相同的维度空间，即 heads * out_channels。这是为了使得边特征能够与节点特征有效地结合。
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            # 最后，将经过线性变换后的边特征加到节点特征上。这一步相当于将边的信息融入到节点特征更新中，丰富了节点特征的表达能力。
            x = x + edge_attr

        # 引入非线性，使得模型可以表达更复杂的模式。
        x = F.leaky_relu(x, self.negative_slope)
        # 计算注意力得分 alpha。具体来说，将激活后的 x 与注意力参数 self.att 相乘，然后在最后一个维度上求和（即，计算每个头部的内积）。
        # 除以 math.sqrt(self.out_channels) 是为了进行缩放（scale），这通常是为了稳定训练过程。
        alpha = (x * self.att).sum(dim=-1) / math.sqrt(self.out_channels)
        # Softmax 函数将注意力得分 alpha 转换为概率分布，这意味着所有邻接节点的注意力得分将会归一化为和为1。这样做的目的是为了更好地解释不同邻接节点对目标节点的影响程度。
        alpha = softmax(alpha, index, ptr, size_i)
        # 这里将 alpha 存储到类的属性 _alpha 中，以便后续可以访问这些注意力权重。这在某些情况下可能用于分析或者调试。
        self._alpha = alpha
        # Dropout 是一种正则化技术，用于防止过拟合。这里将 alpha 中的一部分值随机置为0，以增加模型的鲁棒性。self.dropout 决定了置0的概率。
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
