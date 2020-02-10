const tf = require("@tensorflow/tfjs");

/**
 * Bahdanau's Decoder
 */
class EncoderBahdanau extends tf.layers.Layer {
  constructor(config) {
    super(config || {});

    this.inputVocabSize = config.inputVocabSize;
    this.embeddingDims = config.embeddingDims;
    this.inputLength = config.inputLength;
    this.lstmUnits = config.lstmUnits;
    this.batchSize = config.batchSize;

    // Selects rows from embedding by `encoderInput` values
    this.embedding = tf.layers.embedding({
      inputDim: this.inputVocabSize,
      outputDim: this.embeddingDims,
      inputLength: this.inputLength,
      maskZero: true
    });

    this.LSTM = tf.layers.lstm({
      units: this.lstmUnits,
      returnSequences: true,
      returnState: true
    });
  }

  computeOutputShape(inputShape) {
    return [null, this.inputLength, this.lstm];
  }

  call(input) {
    const encoderOutput = this.embedding.apply(input);

    const lstmOutput = this.LSTM.apply(encoderOutput);

    return { output: lstmOutput[0], lastState: lstmOutput[1] };
  }

  static get className() {
    return "Attention";
  }
}

tf.serialization.registerClass(EncoderBahdanau);
module.exports = EncoderBahdanau;
