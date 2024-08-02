# Smol SAE

Most SAE implementations are either part of research code (messy) or robust code bases (over-engineered).
This sometimes makes it hard to discern the important parts in SAE training. To this end we propose **smol-sae**; a very small, didactic SAE training library. This library deliberately ignores as much as possible while retaining good base performance.

This repo currently contains the base code for the following SAEs:

- Vanilla ([Bricken et al.](https://transformer-circuits.pub/2023/monosemantic-features/index.html))
- Normed ([Conerly et al.](https://transformer-circuits.pub/2024/april-update/index.html))
- Gated ([Rajamanoharan et al.](https://arxiv.org/abs/2404.16014)) **TODO**
- TopK ([Gao et al.](https://arxiv.org/pdf/2406.04093))
- Tokenized ([Dooms et al.](https://openreview.net/forum?id=5Eas7HCe38))

> Please keep in mind that not all code has been thoroughly verified.
