# Efficient Attention: Discussion and Experiments

### Authors
- Sahil Chaudhary (@sahil-chaudhary)
- Mrigank Pawagi (@mrigankpawagi)
- Rohit Jorige (@irenicquasar)
- Jajapuram Nagasai (@jajapuramnagasai)

## Overview
The transformer architecture revolutionized sequence transduction with its novel "Scaled Dot-Product Attention" by outperforming previous recurrent and convolutional models, while dispensing recurrence and convolutions entirely. However, these transformers are prohibitively slow for very long sequences due to their $\mathcal{O}(N^2)$ time complexity in the forward pass, where $N$ is the sequence length. By expressing the self-attention in these transformers as a linear dot-product of kernel feature maps and making use of the associativity property of matrix products can reduce the complexity to $\mathcal{O}(N)$. In addition to these theoretical results, recent empirical results suggest that training of and inference from these models can be made faster and more scalable. One such technique, "Low-Rank Adaption" or LoRA has proven to be particularly valuable in the scalable adaptation of pre-trained models to new tasks through fine-tuning. In this article, we will discuss the results mentioned above. Besides reproducing some of the experiments in these papers, our contributions include new experiments to explore these results.

## Attention Architecture
The self-attention mechanism makes it possible to capture dependencies between different positions in a sequence by simultaneously attending to all positions. The "Scaled Dot-Product Attention" proposed by Vaswani et al.[^1] for an input sequence $x \in \R^{N \times F}$ is computed as $\text{Attention}(x) = \text{softmax}(QK^T/\sqrt{D})V$ where $Q = xW_Q$ (queries), $K = xW_K$ (keys) and $V = xW_V$ (values). Note that $W_Q, W_K \in \R^{F \times D}$ and $W_V \in \R^{F \times F}$ are learned weights. Then the attention vector for the $i$-th position is given by the $i$-th row,

$\text{Attention}(x)_i =\text{softmax}\left(\frac{(QK^T)_i}{\sqrt{D}} \right) V = \text{softmax}\left(\frac{Q_iK^T}{\sqrt{D}} \right) V = \frac{\sum_{j=1}^N \exp(Q_i^TK_j/\sqrt{D})V_j}{\sum_{j=1}^N \exp(Q_i^TK_j/\sqrt{D})}$

Since $\exp(Q_i K_j^T / \sqrt{D})$ has to be determined and stored for every $i, j \in [N]$, this leads to $\mathcal{O}(N^2)$ time and memory complexity (precisely $\mathcal{O}(N^2\max\{D, F\})$).


[^1]: Vaswani

