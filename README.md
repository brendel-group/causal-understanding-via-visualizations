# How Well do Feature Visualizations Support Causal Understanding of CNN Activations?
This repository contains code to reproduce the experiments described in the NeurIPS 2021 paper [How Well do Feature Visualizations Support Causal Understanding of CNN Activations?](https://arxiv.org/abs/2106.12447) by Roland S. Zimmermann*, Judy Borowski*, Robert Geirhos, Matthias Bethge', Tom S. A. Wallis', Wieland Brendel'.
If you have any questions, please reach out via email or create an issue here on GitHub and we'll try to answer it.

## Structure
The [mturk](mturk/README.md) folder contains the implementation of the experiments' UI. Tools to host this on a web server can be found in the [server](server/README.md) directory. To generate the stimuli used in the experiments, look at the [tools/data-generation](tools/data-generation/README.md) folder. For performing the experiment using AWS Mechanical Turk, use the tools proved in [tools/mturk](tools/mturk/README.md). Finally, to evaluate the data and re-create the figures from the paper, use the notebooks provided in [tools/data-analysis](tools/data-analysis/README.md).

## Citation
```bibtex
@article{zimmermann2021well,
  title={How Well do Feature Visualizations Support Causal Understanding of CNN Activations?},
  author={Zimmermann, Roland S and Borowski, Judy and Geirhos, Robert and Bethge, Matthias and Wallis, Thomas SA and Brendel, Wieland},
  journal={arXiv preprint arXiv:2106.12447},
  year={2021}
}
```