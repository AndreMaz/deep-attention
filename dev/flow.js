/**
 * Single passage over all layers with an actual date
 */

const tf = require("@tensorflow/tfjs-node");
const dateFormat = require("../dataset/date_format");

const GetLastTimestepLayer = require("../models/last-time-step-layer");
tf.serialization.registerClass(GetLastTimestepLayer);
const kleur = require("kleur");

const embeddingDims = 64;
const lstmUnits = 64;

const inputVocabSize = 35;
const inputLength = 12;

const outputVocabSize = 13;
const outputLength = 10;

console.log(kleur.bgWhite(kleur.black("INPUT VOCABULARY MAPPINGS")));
console.log(
  kleur.bgWhite(
    kleur.black(`Input Vocal Length : ${dateFormat.INPUT_VOCAB.length}`)
  )
);
dateFormat.INPUT_VOCAB.split("").forEach((value, index) => {
  if (index === 0) {
    console.log(kleur.bgGreen(kleur.black(`Char "\\n": Index ${index}`)));
  } else
    console.log(kleur.bgGreen(kleur.black(`Char "${value}": Index ${index}`)));
});

console.log(kleur.bgWhite(kleur.black("OUTPUT VOCABULARY MAPPINGS")));
console.log(
  kleur.bgWhite(
    kleur.black(`Input Vocal Length : ${dateFormat.OUTPUT_VOCAB.length}`)
  )
);
dateFormat.OUTPUT_VOCAB.split("").forEach((value, index) => {
  if (index === 0) {
    console.log(kleur.bgMagenta(kleur.black(`Char "\\n": Index ${index}`)));
  } else if (index === 1) {
    console.log(kleur.bgMagenta(kleur.black(`Char "\\t": Index ${index}`)));
  } else
    console.log(
      kleur.bgMagenta(kleur.black(`Char "${value}": Index ${index}`))
    );
});

// Original Date = [2019, 10, 1];

// Input Date
const dateAsString = "01.10.2019";
// Input date converted to an array of indices
// [[1, 2, 13, 2, 1, 13, 3, 1, 2, 10, 0, 0],]
const encodeEmbeddingInput = dateFormat.encodeInputDateStrings([dateAsString]);

// Output date
const dateAsOut = "2019-10-01";
// Output date converted to an array of indices
// [[4, 2, 3, 11, 12, 3, 2, 12, 2, 3],]
let decodeEmbeddingInput = dateFormat.encodeOutputDateStrings([dateAsOut]);

// ENCODER //
encodeEmbeddingInput.print();

// Selects rows from embedding by `encoderInput` values
let encoderEmbeddingOutput = tf.layers
  .embedding({
    inputDim: inputVocabSize,
    outputDim: embeddingDims,
    inputLength,
    maskZero: true
  })
  .apply(encodeEmbeddingInput);

console.log(`Shape of encoderEmbeddingOutput`);
console.log(encoderEmbeddingOutput.shape);
encoderEmbeddingOutput.print();

let encoderLSTMOutput = tf.layers
  .lstm({ units: lstmUnits, returnSequences: true }) // Returns all outputs. Including the intermediaries
  // .lstm({ units: lstmUnits }) // Returns only the last output vector.
  .apply(encoderEmbeddingOutput);

console.log(`Shape of encoderLSTMOutput`);
console.log(encoderLSTMOutput.shape);
encoderLSTMOutput.print();

const encoderLast = new GetLastTimestepLayer({ name: "encoderLast" }).apply(
  encoderLSTMOutput
);

console.log(`Shape of encoderLast`);
console.log(encoderLast.shape);
encoderLast.print();

// DECODER //
decodeEmbeddingInput.print();

let decoderEmbeddingOutput = tf.layers
  .embedding({
    inputDim: outputVocabSize,
    outputDim: embeddingDims,
    inputLength: outputLength,
    maskZero: true
  })
  .apply(decodeEmbeddingInput);

console.log(`Shape of decoderEmbeddingOutput`);
console.log(decoderEmbeddingOutput.shape);
decoderEmbeddingOutput.print();

decoderLTSMOutput = tf.layers
  .lstm({ units: lstmUnits, returnSequences: true })
  // .lstm({ units: lstmUnits }) // Returns only the last output vector.
  .apply(decoderEmbeddingOutput, { initialState: [encoderLast, encoderLast] });

console.log(`Shape of decoderLTSMOutput`);
console.log(decoderLTSMOutput.shape);
decoderLTSMOutput.print();

// ATTENTION //
let attentionDot = tf.layers
  .dot({ axes: [2, 2] })
  .apply([decoderLTSMOutput, encoderLSTMOutput]);
console.log(`Shape of attentionDot`);
console.log(attentionDot.shape);
attentionDot.print();

let attentionSoftMax = tf.layers
  .activation({ activation: "softmax", name: "attention" })
  .apply(attentionDot);
console.log(`Shape of attentionSoftMax`);
console.log(attentionSoftMax.shape);
attentionSoftMax.print();

const context = tf.layers
  .dot({ axes: [2, 1], name: "context" })
  .apply([attentionSoftMax, encoderLSTMOutput]);
console.log(`Shape of context`);
console.log(context.shape);
context.print();

const decoderCombinedContext = tf.layers
  .concatenate()
  .apply([context, decoderLTSMOutput]);
console.log(`Shape of decoderCombinedContext`);
console.log(decoderCombinedContext.shape);
decoderCombinedContext.print();

let timeDistributedOutput = tf.layers
  .timeDistributed({
    layer: tf.layers.dense({ units: lstmUnits, activation: "tanh" })
  })
  .apply(decoderCombinedContext);
console.log(`Shape of timeDistributedOutput`);
console.log(timeDistributedOutput.shape);
timeDistributedOutput.print();

let output = tf.layers
  .timeDistributed({
    layer: tf.layers.dense({ units: outputVocabSize, activation: "softmax" })
  })
  .apply(timeDistributedOutput);

console.log(`Shape of output`);
console.log(output.shape);
output.print();
