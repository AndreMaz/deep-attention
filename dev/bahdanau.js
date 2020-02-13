const tf = require("@tensorflow/tfjs-node");
const dateFormat = require("../dataset/date_format");

const { oneStepAttention } = require("../models/bahdanau-attention");

const embeddingDims = 64;
const lstmUnits = 64;

const inputVocabSize = 35;
const inputLength = 12;

const outputVocabSize = 13;
const outputLength = 10;

const batchSize = 128;

// Input Date
const dateAsString = "01.10.2019";
const encodeEmbeddingInput = dateFormat.encodeInputDateStrings([dateAsString]);

// Output date
const dateAsOut = "2019-10-01";
let decodeEmbeddingInput = dateFormat
  .encodeOutputDateStrings([dateAsOut])
  .asType("float32");

let shiftedDecodeEmbeddingInput = tf.concat(
  [
    tf.ones([decodeEmbeddingInput.shape[0], 1]).mul(dateFormat.START_CODE),
    decodeEmbeddingInput.slice(
      [0, 0],
      [decodeEmbeddingInput.shape[0], decodeEmbeddingInput.shape[1] - 1]
    )
  ],
  1
);

encodeEmbeddingInput.print();
decodeEmbeddingInput.print();

// Selects rows from embedding by `encoderInput` values
let encoderEmbeddingOutput = tf.layers
  .embedding({
    inputDim: inputVocabSize,
    outputDim: embeddingDims,
    inputLength,
    maskZero: true
  })
  .apply(encodeEmbeddingInput);
