const tf = require("@tensorflow/tfjs");

/**
 * @typedef {import('@tensorflow/tfjs').Tensor} Tensor
 */

/**
 * Bahdanau's Decoder
 */
class AttentionBahdanau extends tf.layers.Layer {
  constructor(config) {
    super(config || {});

    this.units = config.units;

    this.W1 = tf.layers.dense({ units: this.units });
    this.W2 = tf.layers.dense({ units: this.units });
    this.V = tf.layers.dense({ units: 1 });
  }

  computeOutputShape(inputShape) {
    return inputShape;
  }

  /**
   * @param {Tensor[]} input 1st element is last hidden state; 2nd element is all encoder's hidden states
   * @returns Returns Context Vector and the Attention Weights
   */
  call(input) {
    return tf.tidy(() => {
      /** @type {Tensor} Decoder's last hidden state */
      const decodersPrevHidden = input[0];
      /** @type {Tensor} Encoder's hidden states */
      const encoderOutput = input[1];

      console.log(`Query Shape ${decodersPrevHidden.shape}`);
      console.log(`Values Shape ${encoderOutput.shape}`);

      // hidden shape == (batch_size, hidden size)
      // hidden_with_time_axis shape == (batch_size, 1, hidden size)
      // we are doing this to perform addition to calculate the score
      let hidden_with_time_axis = tf.expandDims(decodersPrevHidden, 1);

      console.log(`hidden_with_time_axis Shape ${hidden_with_time_axis.shape}`);
      // hidden_with_time_axis.print();

      // score shape == (batch_size, max_length, 1)
      // we get 1 at the last axis because we are applying score to self.V
      // the shape of the tensor before applying self.V is (batch_size, max_length, units)
      // v^t * tanh(W1 * decoder's_prev_state + W2 * encoder's_states)
      let score = this.V.apply(
        tf.tanh(
          tf.add(
            this.W2.apply(hidden_with_time_axis),
            this.W1.apply(encoderOutput)
          )
        )
      );

      console.log(`score Shape ${score.shape}`);
      const scoreShape = score.shape;
      score = score.flatten();

      // attention_weights shape == (batch_size, max_length, 1)
      // C++ Backend doesn't support `axis`
      let attentionWeights = tf.softmax(score /* { axis: 1 } */);

      // Reshape back into the original shape
      attentionWeights = attentionWeights.reshape(scoreShape);

      console.log(`attention_weights Shape ${attentionWeights.shape}`);
      console.log(`values Shape ${encoderOutput.shape}`);

      // context_vector shape after sum == (batch_size, hidden_size)
      // context_vector = attention_weights * values;
      // context_vector = tf.reduce_sum(context_vector, (axis = 1));
      let contextVector = tf.mul(attentionWeights, encoderOutput);
      contextVector = tf.sum(contextVector, 1);

      return { contextVector, attentionWeights };
    });
  }

  static get className() {
    return "Attention";
  }
}

tf.serialization.registerClass(AttentionBahdanau);
module.exports = AttentionBahdanau;
