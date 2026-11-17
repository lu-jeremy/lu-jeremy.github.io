---
title: "Sticky Jump Diffusions: A Unifying View of Masked, Continuous, and Hybrid Diffusion"
collection: publications
category: conferences
permalink: /publication/2026-sticky-jump-diffusions
date: 2026-07-12
venue: 'Conference on Neural Information Processing Systems (NeurIPS)'
venue_short: 'NeurIPS 2026'
authors: 'Pascal Jutras-Dubé, Patrick Pynadath, <b>Jeremy Lu</b>, Yuan Gao, Ruqi Zhang'
selected: true
codeurl: 'https://github.com/PascalJD/sticky-jump-diffusions'
thumbnail: 'sticky-jump-diffusions.png'
paperurl: 'https://arxiv.org/abs/2607.10951'
citation: 'P. Jutras-Dubé, P. Pynadath, <b>J. Lu</b>, Y. Gao, and R. Zhang. (2026). &quot;Sticky Jump Diffusions: A Unifying View of Masked, Continuous, and Hybrid Diffusion.&quot; <i>Conference on Neural Information Processing Systems (NeurIPS)</i>.'
excerpt: "Sticky Jump Diffusions (SJDs) are continuous-time Markov processes on ℝᵈ whose discrete anchors are token embeddings. In the forward process, anchors release mass at a hazard rate that diffuses in continuous space; time reversal pairs a score-driven SDE with a sticky jump kernel governed by flux balance. Scores and reverse hazards are estimated through Denoising Hazard Matching using simulation-free cross-entropy training. The framework unifies masked, continuous, and hybrid diffusion as limiting cases and explains features each treats as given. Beyond these limits, the unsticking kernel becomes a design space: cross-position blending corrupts each position toward a blend of its neighbors' clean values or embeddings, improving results on CIFAR-10, Text8, and Sudoku."
---

(**Accepted at NeurIPS 2026**)

[arXiv pre-print](https://arxiv.org/abs/2607.10951)
