const tf = require("@tensorflow/tfjs-node");
const kleur = require("kleur");

/**
 * Examples of TimeDistributed Layer
 * More info: https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/
 */

async function one2one() {
  console.log(kleur.bgGreen(kleur.black("one2one")));
  let xInput = tf.tensor([0.0, 0.2, 0.4, 0.6, 0.8]);
  let yExpected = tf.tensor([0.1, 0.3, 0.5, 0.7, 0.9]);

  // Reshape
  xInput = xInput.reshape([5, 1, 1]);
  // xInput.print();
  console.log(xInput.shape);

  yExpected = yExpected.reshape([5, 1]);
  //  yExpected.print();
  console.log(yExpected.shape);

  const length = 5;

  const numNeurons = length;
  const numBatch = length;
  const numEpoch = 1000;

  let model = tf.sequential();

  model.add(
    tf.layers.lstm({
      units: length,
      inputShape: [1, 1]
    })
  );

  model.add(tf.layers.dense({ units: 1 }));

  model.compile({ loss: "meanSquaredError", optimizer: "adam" });

  await model.fit(xInput, yExpected, {
    epochs: numEpoch,
    verbose: 0
  });

  const result = model.predict(
    tf.tensor3d([0.0, 0.2, 0.4, 0.6, 0.8], [5, 1, 1])
  );
  console.log(kleur.bgMagenta(kleur.black(`one2one prediction`)));
  console.log(result.shape);
  result.print();
  console.log(kleur.bgMagenta(kleur.black(`______________________________`)));
}

one2one()
  .then(res => console.log("one2one - done"))
  .catch(err => console.log(err));

async function many2oneNoTimeDistributed() {
  console.log(kleur.bgGreen(kleur.black("many2oneNoTimeDistributed")));
  let xInput = tf.tensor([0.0, 0.2, 0.4, 0.6, 0.8]);
  let yExpected = tf.tensor([0.1, 0.3, 0.5, 0.7, 0.9]);

  // Reshape
  xInput = xInput.reshape([1, 5, 1]);
  // xInput.print();
  // console.log(xInput.shape);

  yExpected = yExpected.reshape([1, 5]);
  //  yExpected.print();
  // console.log(yExpected.shape);

  const length = 5;

  const numNeurons = length;
  const numBatch = length;
  const numEpoch = 1000;

  let model = tf.sequential();

  model.add(
    tf.layers.lstm({
      units: numNeurons,
      inputShape: [length, 1]
    })
  );

  model.add(tf.layers.dense({ units: length }));

  model.compile({ loss: "meanSquaredError", optimizer: "adam" });

  model.summary();

  await model.fit(xInput, yExpected, {
    epochs: numEpoch,
    verbose: 0
  });

  const result = model.predict(
    tf.tensor3d([0.0, 0.2, 0.4, 0.6, 0.8], [1, 5, 1])
  );
  console.log(
    kleur.bgMagenta(kleur.black(`many2oneNoTimeDistributed prediction`))
  );
  console.log(result.shape);
  result.print();
  console.log(kleur.bgMagenta(kleur.black(`______________________________`)));
}

many2oneNoTimeDistributed()
  .then(res => console.log("many2oneNoTimeDistributed - done"))
  .catch(err => console.log(err));

async function many2oneTimeDistributed() {
  console.log(kleur.bgGreen(kleur.black("many2oneTimeDistributed")));
  let xInput = tf.tensor([0.0, 0.2, 0.4, 0.6, 0.8]);
  let yExpected = tf.tensor([0.1, 0.3, 0.5, 0.7, 0.9]);

  // Reshape
  xInput = xInput.reshape([1, 5, 1]);
  // xInput.print();
  // console.log(xInput.shape);

  yExpected = yExpected.reshape([1, 5, 1]);
  // yExpected.print();
  // console.log(yExpected.shape);

  const length = 5;

  const numNeurons = length;
  const numBatch = length;
  const numEpoch = 1000;

  let model = tf.sequential();

  model.add(
    tf.layers.lstm({
      units: numNeurons,
      inputShape: [length, 1],
      returnSequences: true
    })
  );

  model.add(
    tf.layers.timeDistributed({
      layer: tf.layers.dense({ units: 1 })
    })
  );

  model.compile({ loss: "meanSquaredError", optimizer: "adam" });

  model.summary();

  await model.fit(xInput, yExpected, {
    epochs: numEpoch,
    verbose: 0
  });

  const result = model.predict(
    tf.tensor3d([0.0, 0.2, 0.4, 0.6, 0.8], [1, 5, 1])
  );
  console.log(
    kleur.bgMagenta(kleur.black(`many2oneTimeDistributed prediction`))
  );
  console.log(result.shape);
  result.print();
  console.log(kleur.bgMagenta(kleur.black(`______________________________`)));
}

many2oneTimeDistributed()
  .then(res => console.log("many2oneTimeDistributed - done"))
  .catch(err => console.log(err));

async function many2oneTimeDense() {
  console.log(kleur.bgGreen(kleur.black("many2oneTimeDense")));
  let xInput = tf.tensor([0.0, 0.2, 0.4, 0.6, 0.8]);
  let yExpected = tf.tensor([0.1, 0.3, 0.5, 0.7, 0.9]);

  // Reshape
  xInput = xInput.reshape([1, 5, 1]);
  // xInput.print();
  // console.log(xInput.shape);

  yExpected = yExpected.reshape([1, 5, 1]);
  // yExpected.print();
  // console.log(yExpected.shape);

  const length = 5;

  const numNeurons = length;
  const numBatch = length;
  const numEpoch = 1000;

  let model = tf.sequential();

  model.add(
    tf.layers.lstm({
      units: numNeurons,
      inputShape: [length, 1],
      returnSequences: true
    })
  );

  model.add(tf.layers.dense({ units: 1, inputShape: [5, 5] }));

  model.compile({ loss: "meanSquaredError", optimizer: "adam" });

  model.summary();

  await model.fit(xInput, yExpected, {
    epochs: numEpoch,
    verbose: 0
  });

  const result = model.predict(
    tf.tensor3d([0.0, 0.2, 0.4, 0.6, 0.8], [1, 5, 1])
  );
  console.log(kleur.bgMagenta(kleur.black(`many2oneTimeDense prediction`)));
  console.log(result.shape);
  result.print();
  console.log(kleur.bgMagenta(kleur.black(`______________________________`)));
}

many2oneTimeDense()
  .then(res => console.log("many2oneTimeDense - done"))
  .catch(err => console.log(err));
