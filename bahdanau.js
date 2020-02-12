const tf = require("@tensorflow/tfjs-node");
const { generateDataSet } = require("./dataset/generator");

const EncoderBahdanau = require("./models/bahdanau-attention/encoder");
const DecoderBahdanau = require("./models/bahdanau-attention/decoder");

const configs = {
  dataset: {
    minYear: 1950,
    maxYear: 2050,
    /** Percentage of data used for training */
    trainSplit: 0.25,
    /** Percentage of data used for validation */
    valSplit: 0.15
  },
  args: {
    epochs: 2,
    batchSize: 128,
    learningRate: 0.005
  }
};

async function main() {
  // Generate datasets
  const {
    trainEncoderInput,
    trainDecoderInput,
    trainDecoderOutput,
    valEncoderInput,
    valDecoderInput,
    valDecoderOutput,
    testDateTuples
  } = generateDataSet(configs.dataset);

  let encoderInput = trainEncoderInput.unstack();
  let decoderInput = trainDecoderInput.unstack();
  let decoderOutput = trainDecoderOutput.unstack();

  let arr = [];
  for (let i = 0; i < encoderInput.length; i++) {
    arr[i] = {};
    arr[i].encoderInput = encoderInput[0];
    arr[i].decoderInput = decoderInput[0];
    arr[i].decoderOutput = decoderOutput[0];
  }

  //   const dataset = tf.data.array

  // const optimizer = tf.train.adam(configs.args.learningRate);
  // const loss = tf.metrics.sparseCategoricalAccuracy();
}

function computeLoss(real, pred) {
  let mask = tf.logicalNot(tf.equal(real, 0));
}

function trainer() {}

function trainStep() {}

main();
