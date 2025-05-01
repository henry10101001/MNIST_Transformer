# MNIST Transformer

**Experimenting with generative decoder-only transformers.** I wanted to do something more exciting than just classification, so I made a transformer to generate MNIST digits.

I've been switching between developing on my M3 Macbook Air and training on my RTX 4070. The M3 using mps (Metal Performance Shaders) is surprisingly fast, but the RTX 4070 on cuda is at least 2x faster.

## Results
- I first made a 6.9 million parameter model that performed pretty well. It wasn't fast, but the digits were recognizable.
- I then went overboard with "improvements" and made a 57 million parameter model that trained overnight. I switched out the learned positional embeddings with RoPE, and I switched to a Flash Attention implementation. It was much slower, and the digits were much worse (generated digits had weird slanting and issues with local consistency). I assume this was due to the RoPE, which I didn't fully understand enough to implement correctly (or it's just not suited for this task). From here, I decided to roll back to the simple learned positional embeddings, but I kept flash attention.
- Next, I made a very solid 3.5 million parameter model that was just as good as the original 6.9 million parameter model. It's a bit faster, but the quality could be better. I'm looking to now make sure the flash attention is working right, and then I'll try to refine the model in other ways.

## **Next Fixes** (short term)
- Repository improvements
  - add 3.5 million parameter model to results
  - add loss plots to README
  - start logging models' performance, details, speeds, training times, and training details to a log file
  - organize the generated images into folders by model; add each model's samples to the README
- Make sure **Flash Attention** is implemented correctly.
- Implement **patching** for a huge speed boost (at the cost of quality)
- Better positional encoding
- Optimizations for inference speed
  - smallest model that maintains quality

## **Future Experiments** (long term)
- Models to try:
  - [VAR](https://arxiv.org/abs/2404.02905)
    - next-scale prediction
    - simplify by using [VQ-VAE](https://arxiv.org/abs/1711.00937) instead of VAR's use of VQ-GAN
  - [MAR](https://arxiv.org/abs/2406.11838)
    - throws out the vector quantization
  - [Diffusion](https://arxiv.org/abs/2006.11239)
- Other experiments:
  - Use my model to generate a 200,000 sample synthetic dataset
    - train a second model on the best 100,000 samples of this dataset
    - is more data better? is better quality data better?
  - Add natural language prompting
    - combine MNIST generator with a pretrained LLM or train a single model from scratch?
    - synthetic data generation using gpt-4.1-nano
      - "Draw a 3"
      - "Show me a 7"
      - "What's a 9 look like?"
      - "Show me 2 + four"
      - "Imagine one plus three"
  - Inpainting
  - Move up to CIFAR-10 instead of MNIST

## Acknowledgments

- Portions of this code (`mnist_transformer.py`) are adapted from
  [Andrej Karpathy's nanogpt-lecture](https://github.com/karpathy/ng-video-lecture),
  which is MIT-licensed. This was my primary resource for understanding the basic structure of a transformer and how to implement it in PyTorch. See [`third_party/LICENSE-karpathy-ng-lecture.txt`](third_party/LICENSE-karpathy-ng-lecture.txt) for the original license.