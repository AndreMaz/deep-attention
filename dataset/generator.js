import * as tf from "@tensorflow/tfjs";
import * as dateFormat from "./date_format";

/**
 * Generates the dataset for training, validation and testing
 *
 * @param {object} configs
 * @param {number} configs.minYear Start Year that will be used to generate the dataset
 * @param {number} configs.maxYear End Year that will be used to generate the dataset
 * @param {number} configs.trainSplit Percentage of data used for training
 * @param {number} configs.valSplit Percentage of data used for validation
 */
export function generateDataSet(configs) {
  // Generate ordered data sets
  const dateTuples = generateOrderedDates(configs.minYear, configs.maxYear);

  tf.util.shuffle(dateTuples);

  const numTrain = Math.floor(dateTuples.length * configs.trainSplit);
  const numValidation = Math.floor(dateTuples.length * configs.valSplit);
  console.log(`Number of dates used for training: ${numTrain}`);
  console.log(`Number of dates used for validation: ${numValidation}`);
  console.log(
    `Number of dates used for testing: ` +
      `${dateTuples.length - numTrain - numValidation}`
  );

  const {
    encoderInput: trainEncoderInput,
    decoderInput: trainDecoderInput,
    decoderOutput: trainDecoderOutput
  } = dateTuplesToTensor(dateTuples.slice(0, numTrain));
  const {
    encoderInput: valEncoderInput,
    decoderInput: valDecoderInput,
    decoderOutput: valDecoderOutput
  } = dateTuplesToTensor(dateTuples.slice(numTrain, numTrain + numValidation));

  const testDateTuples = dateTuples.slice(
    numTrain + numValidation,
    dateTuples.length
  );

  return {
    trainEncoderInput,
    trainDecoderInput,
    trainDecoderOutput,
    valEncoderInput,
    valDecoderInput,
    valDecoderOutput,
    testDateTuples
  };
}

/**
 * Generates an ordered array of dates.
 *
 * @param {number} minYear
 * @param {number} maxYear
 *
 * @returns {number[][]} Format: [[2000, 1, 30], ..., [2020, 12, 29]]
 */
function generateOrderedDates(minYear, maxYear) {
  tf.util.assert(
    minYear < maxYear,
    `minYear "${minYear}" is bigger than maxYear ${maxYear}`
  );

  const dateTuples = [];
  for (
    let date = new Date(minYear, 0, 1);
    date.getFullYear() < maxYear;
    date.setDate(date.getDate() + 1)
  ) {
    let entry = [date.getFullYear(), date.getMonth() + 1, date.getDate()];
    dateTuples.push(entry);
  }

  return dateTuples;
}

/**
 *
 * @param {number[][]} dateTuples
 */
function dateTuplesToTensor(dateTuples) {
  return tf.tidy(() => {
    // Convert tuples to supported formats strings.
    // Result:
    // [
    //   [x1_formatA, x2_formatA, ..., x_n_formatA],
    //   [x1_formatB, x2_formatB, ..., x_n_formatB],
    // ]
    const inputs = dateFormat.INPUT_FNS.map(fn =>
      dateTuples.map(tuple => fn(tuple))
    );
    // Flatten arrays into a single one
    const inputStrings = [];
    inputs.forEach(inputs => inputStrings.push(...inputs));

    const encoderInput = dateFormat.encodeInputDateStrings(inputStrings);
    const trainTargetStrings = dateTuples.map(tuple =>
      dateFormat.dateTupleToYYYYDashMMDashDD(tuple)
    );

    let decoderInput = dateFormat
      .encodeOutputDateStrings(trainTargetStrings)
      .asType("float32");

    // One-step time shift: The decoder input is shifted to the left by
    // one time step with respect to the encoder input. This accounts for
    // the step-by-step decoding that happens during inference time.
    decoderInput = tf
      .concat(
        [
          tf.ones([decoderInput.shape[0], 1]).mul(dateFormat.START_CODE),
          decoderInput.slice(
            [0, 0],
            [decoderInput.shape[0], decoderInput.shape[1] - 1]
          )
        ],
        1
      )
      .tile([dateFormat.INPUT_FNS.length, 1]);
    const decoderOutput = tf
      .oneHot(
        dateFormat.encodeOutputDateStrings(trainTargetStrings),
        dateFormat.OUTPUT_VOCAB.length
      )
      .tile([dateFormat.INPUT_FNS.length, 1, 1]);
    return { encoderInput, decoderInput, decoderOutput };
  });
}
