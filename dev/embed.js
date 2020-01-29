const tf = require("@tensorflow/tfjs-node");

const inputVocabSize = 5;
const embeddingDims = 5;
const inputLength = 3;

const embeddingInput = tf.tensor2d([0, 1, 1], [1, 3]);

embeddingInput.print();

// Selects rows from embedding by `encoderInput` values
let embeddingOutput = tf.layers
  .embedding({
    inputDim: inputVocabSize,
    outputDim: embeddingDims,
    inputLength,
    maskZero: true
  })
  .apply(embeddingInput);

embeddingOutput.print();
