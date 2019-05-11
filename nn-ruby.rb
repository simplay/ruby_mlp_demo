# frozen_string_literal: true

FILE = File.realpath(__FILE__)
$LOAD_PATH.unshift File.expand_path('../lib', FILE)

require 'vector'
require 'neuron'
require 'neural_network'

data = [
  Vector[-2, -1],
  Vector[25, 6],
  Vector[17, 4],
  Vector[-15, -6]
]

y_trues = Vector[1, 0, 0, 1]

network = NeuralNetwork.new
network.train(data: data, y_trues: y_trues)
require 'pry'; binding.pry
puts
