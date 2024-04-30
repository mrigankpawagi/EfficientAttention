# Efficient Attention: Discussion and Experiments

### Authors
- [Sahil Chaudhary](https://github.com/sahil-chaudhary)
- [Mrigank Pawagi](https://github.com/mrigankpawagi)
- [Rohit Jorige](https://github.com/irenicquasar)
- [Jajapuram Nagasai](https://github.com/jajapuramnagasai)

## Overview
The transformer architecture revolutionized sequence transduction with its novel "Scaled Dot-Product Attention" by outperforming previous recurrent and convolutional models, while dispensing recurrence and convolutions entirely. However, these transformers are prohibitively slow for very long sequences due to their $\mathcal{O}(N^2)$ time complexity in the forward pass, where $N$ is the sequence length. By expressing the self-attention in these transformers as a linear dot-product of kernel feature maps and making use of the associativity property of matrix products can reduce the complexity to $\mathcal{O}(N)$. In addition to these theoretical results, recent empirical results suggest that training of and inference from these models can be made faster and more scalable. One such technique, "Low-Rank Adaption" or LoRA has proven to be particularly valuable in the scalable adaptation of pre-trained models to new tasks through fine-tuning. In this article, we will discuss the results mentioned above. Besides reproducing some of the experiments in these papers, our contributions include new experiments to explore these results.

## Attention Architecture
The self-attention mechanism makes it possible to capture dependencies between different positions in a sequence by simultaneously attending to all positions. The "Scaled Dot-Product Attention" proposed by Vaswani et al.[^1] for an input sequence $x \in \mathbb{R}^{N \times F}$ is computed as $\text{Attention}(x) = \text{softmax}(QK^T/\sqrt{D})V$ where $Q = xW_Q$ (queries), $K = xW_K$ (keys) and $V = xW_V$ (values). Note that $W_Q, W_K \in \mathbb{R}^{F \times D}$ and $W_V \in \mathbb{R}^{F \times F}$ are learned weights. Then the attention vector for the $i$-th position is given by the $i$-th row,

```math
\text{Attention}(x)_i =\text{softmax}\left(\frac{(QK^T)_i}{\sqrt{D}} \right) V = \text{softmax}\left(\frac{Q_iK^T}{\sqrt{D}} \right) V = \frac{\sum\limits_{j=1}^N \exp(Q_i^TK_j/\sqrt{D})V_j}{\sum\limits_{j=1}^N \exp(Q_i^TK_j/\sqrt{D})}
```

Since $\exp(Q_i K_j^T / \sqrt{D})$ has to be determined and stored for every $i, j \in [N]$, this leads to $\mathcal{O}(N^2)$ time and memory complexity (precisely $\mathcal{O}(N^2\max\{D, F\})$<i></i>).

## Towards Efficient Attention - Linearized Attention
The exponential similarity function above can be replaced with any map $\text{sim}: \mathbb{R}^D \times \mathbb{R}^D \to \mathbb{R}_+$. This includes all kernels between the two spaces. Katharopoulos et al.[^2] show that given such a kernel with a feature representation $\phi$, we can rewrite the attention computation as

```math
\text{Attention}(x)_i = \frac{\sum\limits_{j=1}^N \phi(Q_i)^T\phi(K_j)V_j}{\sum\limits_{j=1}^N \phi(Q_i)^T\phi(K_j)} = \frac{\phi(Q_i)^T\sum\limits_{j=1}^N \phi(K_j)V_j^T}{ \phi(Q_i)^T \sum\limits_{j=1}^N \phi(K_j)}
```

Since $\sum\limits_{j=1}^N \phi(K_j)V_j^T$ and $\sum\limits_{j=1}^N \phi(K_j)$ can be computed once reused for every query, this formulation has $\mathcal{O}(N)$ time and memory complexity (precisely $\mathcal{O}(NCF)$ where $C$ is the dimension of the range space of the feature map).

**References**

[^1]: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, _Advances in Neural Information Processing Systems_, volume 30. Curran Associates, Inc., 2017.

[^2]: Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are RNNs: Fast Autoregressive Transformers With Linear Attention. In _Proceedings of the 37th International Conference on Machine Learning_, ICML’20. JMLR.org, 2020.

