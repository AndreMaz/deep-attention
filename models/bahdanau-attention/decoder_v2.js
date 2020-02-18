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

    this.cell = tf.layers.lstmCell({
      units: this.lstmUnits
    });
    this.cell.build([this.lstmUnits]);

    // Selects rows from embedding by `encoderInput` values
    this.embedding = tf.layers.embedding({
      inputDim: this.outputVocabSize,
      outputDim: this.embeddingDims,
      // inputLength: this.outputLength,
      maskZero: true
    });

    /*
    this.LSTM = tf.layers.lstm({
      units: this.lstmUnits,
      returnSequences: true,
      returnState: true
    });

    this.fc = tf.layers.dense({ units: this.outputVocabSize });
    */

    this.attention = new AttentionBahdanau({
      name: "attention",
      units: this.embeddingDims
    });
  }

  computeOutputShape(inputShape) {
    let outputShape = inputShape[0];

    return [...outputShape, this.lstmUnits];
  }

  call(input) {
    return tf.tidy(() => {
      /** @type {Tensor} Decoder's Input */
      let x = input[0];

      /** @type {Tensor} Decoder's Last Hidden State */
      let decodersPrevHidden = input[1];

      /** @type {Tensor} Encoder's hidden states */
      const encoderOutput = input[2];

      /*
      const { contextVector, attentionWeights } = this.attention.apply([
        decodersPrevHidden,
        encoderOutput
      ]);

      let embeddingOutput = this.embedding.apply(x);

      x = tf.concat([tf.expandDims(contextVector, 1), x], -1);
      */

      let perStepInputs = x.unstack(1);
      let perStepOutputs = [];
      const timeSteps = perStepInputs.length;
      let lastOutput;

      let hMinus1 = decodersPrevHidden;
      let cMinus1 = decodersPrevHidden;

      for (let t = 0; t < timeSteps; ++t) {
        let currentInput = perStepInputs[t];

        /*
        const { contextVector, attentionWeights } = this.attention.apply([
          stateMinus1,
          encoderOutput
        ]);*/

        currentInput = this.embedding.apply(currentInput);

        /*currentInput = tf.concat(
          [tf.expandDims(contextVector, 1), currentInput],
          -1
        );*/
        // currentInput = tf.concat([contextVector, currentInput], -1);

        let stepOutputs = this.cell.call([currentInput, hMinus1, cMinus1], {});

        lastOutput = stepOutputs[0];

        hMinus1 = stepOutputs[1];
        cMinus1 = stepOutputs[2];

        perStepOutputs.push(lastOutput);

        // currentInput.print();
      }

      let output = tf.stack(perStepOutputs, 1);

      return output;
    });
  }

  static get className() {
    return "Decoder";
  }
}

tf.serialization.registerClass(DecoderBahdanau);
module.exports = DecoderBahdanau;
