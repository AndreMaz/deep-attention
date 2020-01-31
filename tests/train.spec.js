require("@tensorflow/tfjs-node");
const dateFormat = require("../dataset/date_format");
const { generateDataSet } = require("../dataset/generator");

const configs = {
  dataset: {
    minYear: 2049,
    maxYear: 2050,
    trainSplit: 0.25,
    valSplit: 0.15
  }
};

describe("generateBatchesForTraining", () => {
  it("generateDataForTraining", () => {
    const {
      trainEncoderInput,
      trainDecoderInput,
      trainDecoderOutput,
      valEncoderInput,
      valDecoderInput,
      valDecoderOutput,
      testDateTuples
    } = generateDataSet(configs.dataset);

    const numTrain = trainEncoderInput.shape[0];
    const numVal = valEncoderInput.shape[0];

    /*
    expect(numTrain / numVal).toBeCloseTo(2);
    expect(trainEncoderInput.shape).toEqual([
      numTrain,
      dateFormat.INPUT_LENGTH
    ]);
    expect(trainDecoderInput.shape).toEqual([
      numTrain,
      dateFormat.OUTPUT_LENGTH
    ]);
    expect(trainDecoderOutput.shape).toEqual([
      numTrain,
      dateFormat.OUTPUT_LENGTH,
      dateFormat.OUTPUT_VOCAB.length
    ]);
    expect(valEncoderInput.shape).toEqual([numVal, dateFormat.INPUT_LENGTH]);
    expect(valDecoderInput.shape).toEqual([numVal, dateFormat.OUTPUT_LENGTH]);
    expect(valDecoderOutput.shape).toEqual([
      numVal,
      dateFormat.OUTPUT_LENGTH,
      dateFormat.OUTPUT_VOCAB.length
    ]);
    expect(testDateTuples[0].length).toEqual(3);
    expect(testDateTuples[testDateTuples.length - 1].length).toEqual(3);
    */
  });
});
