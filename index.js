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

function printShapes(opts) {
  const {
    trainEncoderInput,
    trainDecoderInput,
    trainDecoderOutput,
    valEncoderInput,
    valDecoderInput,
    valDecoderOutput,
    testDateTuples
  } = opts;

  console.log(`trainEncoderInput Shape`);
  console.log(trainEncoderInput.shape);
  console.log(`_______________________`);

  console.log(`trainDecoderInput Shape`);
  console.log(trainDecoderInput.shape);
  console.log(`_______________________`);

  console.log(`trainDecoderOutput Shape`);
  console.log(trainDecoderOutput.shape);
  console.log(`_______________________`);

  console.log(`valEncoderInput Shape`);
  console.log(valEncoderInput.shape);
  console.log(`_______________________`);

  console.log(`valDecoderInput Shape`);
  console.log(valDecoderInput.shape);
  console.log(`_______________________`);

  console.log(`valDecoderOutput Shape`);
  console.log(valDecoderOutput.shape);
  console.log(`_______________________`);
}

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

  printShapes({
    trainEncoderInput,
    trainDecoderInput,
    trainDecoderOutput,
    valEncoderInput,
    valDecoderInput,
    valDecoderOutput,
    testDateTuples
  });

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
