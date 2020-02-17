const tf = require("@tensorflow/tfjs-node");
const shelljs = require("shelljs");
const fs = require("fs");
const { generateDataSet } = require("./dataset/generator");

const { modelFactory } = require("./models");

const { runSeq2SeqInference } = require("./models/inference");
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
    batchSize: 128,
    savePath: "./out/model"
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

  const createModel = modelFactory("bahdanau");
  const model = createModel(
    dateFormat.INPUT_VOCAB.length,
    dateFormat.OUTPUT_VOCAB.length,
    dateFormat.INPUT_LENGTH,
    dateFormat.OUTPUT_LENGTH,
    "general"
  );
  model.summary();

  // Train the model
  await model.fit([trainEncoderInput, trainDecoderInput], trainDecoderOutput, {
    epochs: configs.args.epochs,
    batchSize: configs.args.batchSize,
    shuffle: true,
    validationData: [[valEncoderInput, valDecoderInput], valDecoderOutput]
  });

  /*
  // Save the model.
  if (configs.args.savePath != null && configs.args.savePath.length) {
    if (!fs.existsSync(configs.args.savePath)) {
      shelljs.mkdir("-p", configs.args.savePath);
    }
    const saveURL = `file://${configs.args.savePath}`;
    await model.save(saveURL);
    console.log(`Saved model to ${saveURL}`);
  }

  // Run seq2seq inference tests and print the results to console.
  const numTests = 10;
  for (let n = 0; n < numTests; ++n) {
    for (const testInputFn of dateFormat.INPUT_FNS) {
      const inputStr = testInputFn(testDateTuples[n]);
      console.log("\n-----------------------");
      console.log(`Input string: ${inputStr}`);
      const correctAnswer = dateFormat.dateTupleToYYYYDashMMDashDD(
        testDateTuples[n]
      );
      console.log(`Correct answer: ${correctAnswer}`);

      const { outputStr } = await runSeq2SeqInference(model, inputStr);
      const isCorrect = outputStr === correctAnswer;
      console.log(`Model output: ${outputStr} (${isCorrect ? "OK" : "WRONG"})`);
    }
  }
  */
}

main();
