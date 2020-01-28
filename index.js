const tf = require("@tensorflow/tfjs-node");
const { generateDataSet } = require("./dataset/generator");

const configs = {
  dataset: {
    minYear: 1950,
    maxYear: 2050,
    /** Percentage of data used for training */
    trainSplit: 0.25,
    /** Percentage of data used for validation */
    valSplit: 0.15
  }
};

function main() {
  generateDataSet(configs.dataset);
}

main();
