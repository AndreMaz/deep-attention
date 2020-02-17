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

    this.cell = tf.layers.lstmCell({ units: this.lstmUnits });
    this.cell.build([this.lstmUnits]);

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

    /*
    this.attention = new AttentionBahdanau({
      name: "attention",
      units: this.lstmUnits
    });
    */
  }

  computeOutputShape(inputShape) {
    // console.log(inputShape);
    // return inputShape[0];
    return [128, 10, 32];
  }

  call(input) {
    return tf.tidy(() => {
      /** @type {Tensor} Decoder's Input */
      let x = input[0];

      /** @type {Tensor} Decoder's Last Hidden State */
      let initialStates = input[1];

      /** @type {Tensor} Encoder's hidden states */
      const enc_output = input[2];

      let embeddingOutput = this.embedding.apply(x);

      let perStepInputs = embeddingOutput.unstack(1);
      let perStepOutputs = [];
      const timeSteps = perStepInputs.length;
      let lastOutput;

      let stateMinus1 = initialStates;
      for (let t = 0; t < timeSteps; ++t) {
        const currentInput = perStepInputs[t];

        let stepOutputs = this.cell.call(
          [currentInput, stateMinus1, stateMinus1],
          {}
        );

        lastOutput = stepOutputs[0];
        stateMinus1 = stepOutputs[1];

        perStepOutputs.push(lastOutput);

        // currentInput.print();
      }

      // console.log(u);
      // passing the concatenated vector to the LSTM
      // const lstmOutput = this.LSTM.apply(decoderLSTMOutput);

      let o = tf.stack(perStepOutputs, 1);
      // console.log(o.shape);

      return o;
    });
  }

  stepFunction(inputs, states) {
    const outputs = this.cell.call([inputs].concat(states), cellCallKwargs);
    // Marshall the return value into output and new states.
    return [outputs[0], outputs.slice(1)];
  }

  static get className() {
    return "Decoder";
  }
}

tf.serialization.registerClass(DecoderBahdanau);
module.exports = DecoderBahdanau;
