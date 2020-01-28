const tf = require("@tensorflow/tfjs-node");
const { generateDataSet } = require("./dataset/generator");
const { runSeq2SeqInference, createModel } = require("./models/lstm-attention");
const dateFormat = require("./dataset/date_format");

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
    batchSize: 128
  }
};

async function main() {
  const {
    trainEncoderInput,
    trainDecoderInput,
    trainDecoderOutput,
    valEncoderInput,
    valDecoderInput,
    valDecoderOutput,
    testDateTuples
  } = generateDataSet(configs.dataset);

  const model = createModel(
    dateFormat.INPUT_VOCAB.length,
    dateFormat.OUTPUT_VOCAB.length,
    dateFormat.INPUT_LENGTH,
    dateFormat.OUTPUT_LENGTH
  );
  model.summary();

  await model.fit([trainEncoderInput, trainDecoderInput], trainDecoderOutput, {
    epochs: configs.args.epochs,
    batchSize: configs.args.batchSize,
    shuffle: true,
    validationData: [[valEncoderInput, valDecoderInput], valDecoderOutput]
  });
}

main();
