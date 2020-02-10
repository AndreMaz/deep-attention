const kleur = require("kleur");

const Models = {
  seq2seq: require("./seq2seq"),
  luong: require("./luong-attention"),
  bahdanau: require("./bahdanau-attention")
};

/**
 * @param {string} modelType
 */
function modelFactory(modelType) {
  modelType = modelType.toLowerCase();
  if (Object.keys(Models).indexOf(modelType) === -1) {
    throw new Error("Unknown Model");
  }

  console.log(
    `Loaded: ${kleur.bgGreen(
      kleur.black(modelType.toUpperCase())
    )} Model Builder`
  );

  return Models[modelType].createModel;
}

module.exports = { modelFactory };
