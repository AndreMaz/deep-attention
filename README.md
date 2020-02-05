This is a slight refactor of Tensorflow's [data-conversion-attention](https://github.com/tensorflow/tfjs-examples/tree/master/date-conversion-attention) example.
All the credit goes to the TF team and the people that built the model.

I've just refactored things (in a way that make more sense to me) while learning the attention model.

## Changes

- I've reorganized the file structure
- I've dropped the frontend part as I'm only interested in the model
- I'm using Jest instead of Jasmine

## Running

- `npm run train`   - Train the model
- `npm run test`    - Run unit tests
- `npm run flow`    - One execution of the model (with [apply()](https://js.tensorflow.org/api/latest/#tf.layers.Layer.apply)) over an actual input.

## Useful Links
- [Attention Mechanism](https://blog.floydhub.com/attention-mechanism/)
- [Attn: Illustrated Attention](https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)
- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)