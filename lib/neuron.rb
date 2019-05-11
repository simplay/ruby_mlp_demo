# frozen_string_literal: true

class Neuron
  attr_accessor :weights, :bias, :weighted_sum

  # @param weights [Vector]
  # @param bias [Float]
  def initialize(weights:, bias:)
    @weights = weights
    @bias = bias
    @weighted_sum = 0
  end

  # @param inputs [Vector] activations from neurons in previous layer.
  # @return [Float] activation
  def feedforward(inputs:)
    @weighted_sum = weights.dot(inputs) + bias
    sigmoid(weighted_sum)
  end

  def dw(activation)
    activation * derivative_sigmoid(weighted_sum)
  end

  def dh(activation)
    activation * derivative_sigmoid(weighted_sum)
  end

  def db
    derivative_sigmoid(weighted_sum)
  end

  def update_weight(idx:, value:)
    self.weights[idx] -= value
  end

  def update_bias(value:)
    self.bias -= value
  end

  private

  def derivative_sigmoid(x)
    f = sigmoid(x)
    f * (1.0 - f)
  end

  # activation function
  def sigmoid(x)
    1.0 / (1.0 + Math.exp(-x))
  end
end
