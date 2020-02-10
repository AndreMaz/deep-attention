const tf = require("@tensorflow/tfjs");
const dateFormat = require("../dataset/date_format");

/**
 * Perform sequence-to-sequence decoding for date conversion.
 *
 * @param {tf.Model} model The model to be used for the sequence-to-sequence
 *   decoding, with two inputs:
 *   1. Encoder input of shape `[numExamples, inputLength]`
 *   2. Decoder input of shape `[numExamples, outputLength]`
 *   and one output:
 *   1. Decoder softmax probability output of shape
 *      `[numExamples, outputLength, outputVocabularySize]`
 * @param {string} inputStr Input date string to be converted.
 * @return {{outputStr: string, attention?: tf.Tensor}}
 *   - The `outputStr` field is the output date string.
 *   - If and only if `getAttention` is `true`, the `attention` field will
 *     be populated by attention matrix as a `tf.Tensor` of
 *     dtype `float32` and shape `[]`.
 */
async function runSeq2SeqInference(model, inputStr, getAttention = false) {
  return tf.tidy(() => {
    const encoderInput = dateFormat.encodeInputDateStrings([inputStr]);
    const decoderInput = tf.buffer([1, dateFormat.OUTPUT_LENGTH]);
    decoderInput.set(dateFormat.START_CODE, 0, 0);

    for (let i = 1; i < dateFormat.OUTPUT_LENGTH; ++i) {
      const predictOut = model.predict([encoderInput, decoderInput.toTensor()]);

      const output = predictOut.argMax(2).dataSync()[i - 1];
      predictOut.dispose();
      decoderInput.set(output, 0, i);
    }

    const output = { outputStr: "" };

    // The `tf.Model` instance used for the final time step varies depending on
    // whether the attention matrix is requested or not.
    let finalStepModel = model;
    if (getAttention) {
      // If the attention matrix is requested, construct a two-output model.
      // - The 1st output is the original decoder output.
      // - The 2nd output is the attention matrix.
      finalStepModel = tf.model({
        inputs: model.inputs,
        outputs: model.outputs.concat([model.getLayer("attention").output])
      });
    }

    const finalPredictOut = finalStepModel.predict([
      encoderInput,
      decoderInput.toTensor()
    ]);

    let decoderFinalOutput; // The decoder's final output.
    if (getAttention) {
      decoderFinalOutput = finalPredictOut[0];
      output.attention = finalPredictOut[1];
    } else {
      decoderFinalOutput = finalPredictOut;
    }
    decoderFinalOutput = decoderFinalOutput.argMax(2).dataSync()[
      dateFormat.OUTPUT_LENGTH - 1
    ];

    for (let i = 1; i < decoderInput.shape[1]; ++i) {
      output.outputStr += dateFormat.OUTPUT_VOCAB[decoderInput.get(0, i)];
    }
    output.outputStr += dateFormat.OUTPUT_VOCAB[decoderFinalOutput];
    return output;
  });
}

module.exports = { runSeq2SeqInference };
