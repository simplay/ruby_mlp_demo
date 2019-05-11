# frozen_string_literal: true

class NeuralNetwork
  EPOCHS = 1000
  ALPHA = 0.1

  def initialize
    @h1 = Neuron.new(weights: random_weighs, bias: rand)
    @h2 = Neuron.new(weights: random_weighs, bias: rand)
    @o1 = Neuron.new(weights: random_weighs, bias: rand)
  end

  def random_weighs
    Vector.new([rand, rand])
  end

  # activation function
  def sigmoid(x)
    1.0 / (1.0 + Math.exp(-x))
  end

  def derivative_sigmoid(x)
    f = sigmoid(x)
    f * (1.0 - f)
  end

  def mse(y_trues, y_preds)
    values = y_trues.sub(y_preds).map { |i| i ** 2 }
    values.sum / values.length
  end

  # @param inputs [Vector]
  # @return [Float]
  def feedforward(inputs:)
    out_h1 = @h1.feedforward(inputs: inputs)
    out_h2 = @h2.feedforward(inputs: inputs)
    @o1.feedforward(inputs: Vector.new([out_h1, out_h2]))
  end

  # @data [Vector<Vector>>] training data
  # @data [Vector] ground truths
  def train(data:, y_trues:)
    EPOCHS.times do |epoch|
      data.zip(y_trues).each do |x, y_true|
        # do a feedforward
        activation_h1 = @h1.feedforward(inputs: x)
        activation_h2 = @h2.feedforward(inputs: x)
        y_pred = @o1.feedforward(inputs: Vector.new([activation_h1, activation_h2]))

        # calculate partial derivatives
        dL_dypred = -2.0 * (y_true - y_pred)

        # neuron o1
        dypred_dw5 = @o1.dw(activation_h1)
        dypred_dw6 = @o1.dw(activation_h2)
        dypred_db3 = @o1.db

        dypred_dh1 = @o1.weights[0] * derivative_sigmoid(@o1.weighted_sum)
        dypred_dh2 = @o1.weights[1] * derivative_sigmoid(@o1.weighted_sum)

        # neuron h1
        dh1_dw1 = @h1.dw(x[0])
        dh1_dw2 = @h1.dw(x[1])
        dh1_db1 = @h1.db

        # neuron h2
        dh2_dw3 = @h2.dw(x[0])
        dh2_dw4 = @h2.dw(x[1])
        dh2_db2 = @h2.db

        # update weights
        @h1.update_weight(idx: 0, value: ALPHA * dL_dypred * dypred_dh1 * dh1_dw1)
        @h1.update_weight(idx: 1, value: ALPHA * dL_dypred * dypred_dh1 * dh1_dw2)
        @h1.update_bias(value: ALPHA * dL_dypred * dypred_dh1 * dh1_db1)

        @h2.update_weight(idx: 0, value: ALPHA * dL_dypred * dypred_dh2 * dh2_dw3)
        @h2.update_weight(idx: 1, value: ALPHA * dL_dypred * dypred_dh2 * dh2_dw4)
        @h2.update_bias(value: ALPHA * dL_dypred * dypred_dh2 * dh2_db2)

        @o1.update_weight(idx: 0, value: ALPHA * dL_dypred * dypred_dw5)
        @o1.update_weight(idx: 1, value: ALPHA * dL_dypred * dypred_dw6)
        @o1.update_bias(value: ALPHA * dL_dypred * dypred_db3)
      end

      if epoch % 10 == 0
        y_preds = data.map do |d|
          feedforward(inputs: d)
        end
        y_preds = Vector[*y_preds]
        loss = mse(y_trues, y_preds)
        puts "Epoch #{epoch} loss: #{loss}"
      end
    end
  end
end
