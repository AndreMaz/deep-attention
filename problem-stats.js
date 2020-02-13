const tf = require("@tensorflow/tfjs-node");
const dateFormat = require("./dataset/date_format");
const kleur = require("kleur");

console.log(kleur.bgWhite(kleur.black("INPUT VOCABULARY MAPPINGS")));
console.log(
  kleur.bgWhite(
    kleur.black(`Input Vocal Length : ${dateFormat.INPUT_VOCAB.length}`)
  )
);
dateFormat.INPUT_VOCAB.split("").forEach((value, index) => {
  if (index === 0) {
    // "\n" for padding for both input and output
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
    // "\n" for padding for both input and output
    console.log(kleur.bgMagenta(kleur.black(`Char "\\n": Index ${index}`)));
  } else if (index === 1) {
    // "\t" represents start-of-sequence (SOS) token
    console.log(kleur.bgMagenta(kleur.black(`Char "\\t": Index ${index}`)));
  } else
    console.log(
      kleur.bgMagenta(kleur.black(`Char "${value}": Index ${index}`))
    );
});

// Original Date = [2019, 10, 1];
console.log(kleur.bgWhite(kleur.black("POSSIBLE INPUT FORMATS")));
console.log(
  kleur.bgWhite(
    kleur.black(
      `NUMBER OF SUPPORTED DATE FORMATS: ${dateFormat.INPUT_FNS.length}`
    )
  )
);
let inputDate = [[2019, 10, 1]];
inputDate.forEach(date => {
  dateFormat.INPUT_FNS.forEach(fn => {
    console.log(kleur.bgCyan(kleur.black(fn(date))));
  });
});

// Input Date
const dateAsString = "01.10.2019";
// Input date converted to an array of indices
// [[1, 2, 13, 2, 1, 13, 3, 1, 2, 10, 0, 0],]
const encodeEmbeddingInput = dateFormat.encodeInputDateStrings([dateAsString]);

// Output date
const dateAsOut = "2019-10-01";
// Output date converted to an array of indices
// [[4, 2, 3, 11, 12, 3, 2, 12, 2, 3],]
let decodeEmbeddingInput = dateFormat
  .encodeOutputDateStrings([dateAsOut])
  .asType("float32");
// Shift the input
let shiftedDecodeEmbeddingInput = tf.concat(
  [
    tf
      // Creates tensor with all elems set to one
      .ones([decodeEmbeddingInput.shape[0], 1])
      // Element-wise multiplication A * B
      .mul(dateFormat.START_CODE),

    // Removes last column from the encoding
    decodeEmbeddingInput.slice(
      [0, 0], // Start position
      [decodeEmbeddingInput.shape[0], decodeEmbeddingInput.shape[1] - 1] // size of a slice
    )
  ],
  1
);

// ENCODER //
console.log(kleur.bgWhite(kleur.black("ENCODER'S INPUT EXAMPLE: 01.10.2019")));
encodeEmbeddingInput.print();
// DECODER //
console.log(
  kleur.bgWhite(kleur.black("DECODER NOT SHIFTED EXAMPLE: 2019-10-01"))
);
decodeEmbeddingInput.print();
// DECODER //
console.log(
  kleur.bgWhite(kleur.black("DECODER'S SHIFTED INPUT EXAMPLE: 2019-10-01"))
);
shiftedDecodeEmbeddingInput.print();

console.log(kleur.bgWhite(kleur.black("DECODER'S OUTPUT EXAMPLE: 2019-10-01")));
const decoderOutput = tf.oneHot(
  dateFormat.encodeOutputDateStrings(["2019-10-01"]),
  dateFormat.OUTPUT_VOCAB.length
);
decoderOutput.print();
