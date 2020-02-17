const tf = require("@tensorflow/tfjs-node");
const dateFormat = require("../dataset/date_format");

const DecoderBahdanau = require("../models/bahdanau-attention/decoder_v2");
const GetLastTimestepLayer = require("../models/bahdanau-attention/last-time-step-layer");

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

let encoderLSTMOutput = tf.layers
  .lstm({ units: lstmUnits, returnSequences: true, name: "encoderLSTM" }) // `returnSequences` returns all the states. Not only the last one
  .apply(encoderEmbeddingOutput);

let lastEncoderState = new GetLastTimestepLayer({
  name: "lastEncoderState"
}).apply(encoderLSTMOutput);

/*
  const decoderEmbeddingInput = tf.input({
  shape: [outputLength],
  name: "embeddingDecoderInput"
});
*/

let decoderLSTMOutput = new DecoderBahdanau({
  name: "CustomLSTM",
  outputLength: outputLength,
  outputVocabSize: outputVocabSize,
  embeddingDims: lstmUnits,
  lstmUnits: lstmUnits,
  batchSize: 2,
  cell: tf.layers.lstmCell({ units: lstmUnits })
}).apply([shiftedDecodeEmbeddingInput, lastEncoderState, encoderLSTMOutput]);
