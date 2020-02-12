const tf = require("@tensorflow/tfjs-node");
const dateFormat = require("../dataset/date_format");

const DecoderBahdanau = require("../models/bahdanau-attention/decoder");

const GetLastTimestepLayer = require("../models/luong-attention/last-time-step-layer");
tf.serialization.registerClass(GetLastTimestepLayer);

const AttentionBahdanau = require("../models/bahdanau-attention/attention");

const EncoderBahdanau = require("../models/bahdanau-attention/encoder");

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

let encoder = new EncoderBahdanau({
  name: "EncoderBahdanau",
  inputLength,
  inputVocabSize,
  embeddingDims,
  lstmUnits,
  batchSize
});
const hidden = encoder.initHiddenState();
console.log(hidden.shape);

const { output, lastState } = encoder.apply([encodeEmbeddingInput, hidden]);

output.print();
lastState.print();

/*
const { contextVector, attentionWeights } = new AttentionBahdanau({
  name: "attention",
  units: 10
}).apply([lastState, output]);

contextVector.print();
attentionWeights.print();
*/

/*
const { x, state, attentionWeights } = new DecoderBahdanau({
  name: "bahdanau",
  outputLength,
  outputVocabSize,
  embeddingDims,
  lstmUnits,
  batchSize
}).apply([tf.randomUniform([1, 1]), lastState, output]);

console.log(x.shape);
console.log(state.shape);
console.log(attentionWeights.shape);

x.print();
state.print();
attentionWeights.print();
*/
