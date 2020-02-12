const tf = require("@tensorflow/tfjs");
const AttentionBahdanau = require("./attention");

/**
 * @typedef {import('@tensorflow/tfjs').Tensor} Tensor
 */

/**
 * Bahdanau's Decoder
 */
class DecoderBahdanau extends tf.layers.Layer {
  constructor(config) {
    super(config || {});

    this.outputLength = config.outputLength;
    this.outputVocabSize = config.outputVocabSize;
    this.embeddingDims = config.embeddingDims;
    this.lstmUnits = config.lstmUnits;
    this.batchSize = config.batchSize;

    // Selects rows from embedding by `encoderInput` values
    this.embedding = tf.layers.embedding({
      inputDim: this.outputVocabSize,
      outputDim: this.embeddingDims,
      // inputLength: this.outputLength,
      maskZero: true
    });

    this.LSTM = tf.layers.lstm({
      units: this.lstmUnits,
      returnSequences: true,
      returnState: true
    });

    this.fc = tf.layers.dense({ units: this.outputVocabSize });

    this.attention = new AttentionBahdanau({
      name: "attention",
      units: this.lstmUnits
    });
  }

  computeOutputShape(inputShape) {
    return inputShape;
  }

  /**
   * @param {Tensor[]} input
   * @returns
   */
  call(input) {
    /** @type {Tensor} Decoder's Input */
    let x = input[0];
    /** @type {Tensor} Decoder's Last Hidden State */
    const hidden = input[1];
    /** @type {Tensor} Encoder's hidden states */
    const enc_output = input[2];

    // enc_output shape == (batch_size, max_length, hidden_size)
    const { contextVector, attentionWeights } = this.attention.apply([
      hidden,
      enc_output
    ]);

    // x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = this.embedding.apply(x);

    // x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expandDims(contextVector, 1), x], -1);

    // passing the concatenated vector to the LSTM
    const lstmOutput = this.LSTM.apply(x);
    let output = lstmOutput[0];
    let state = lstmOutput[1];

    // output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, [1, output.shape[2]]);

    // output shape == (batch_size, vocab)
    let pred = this.fc.apply(output);

    return { pred, state, attentionWeights };
    // return 1;
  }

  static get className() {
    return "Decoder";
  }
}

tf.serialization.registerClass(DecoderBahdanau);
module.exports = DecoderBahdanau;
