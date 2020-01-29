const tf = require("@tensorflow/tfjs-node");

const GetLastTimestepLayer = require("../models/last-time-step-layer");
tf.serialization.registerClass(GetLastTimestepLayer);

const inputVocabSize = 5;
const embeddingDims = 5;
const lstmUnits = 5;
const inputLength = 3;

// ENCODER //
const encodeEmbeddingInput = tf.tensor2d([0, 1, 1], [1, 3]);
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
const decodeEmbeddingInput = tf.tensor2d([1, 2, 2], [1, 3]);
decodeEmbeddingInput.print();

const outputVocabSize = 4;
const outputLength = 3;

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
