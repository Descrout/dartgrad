import 'const.dart';
import 'value.dart';

class Neuron {
  final List<Value> weights;
  final Value bias;

  Neuron({required int inputLength})
    : assert(inputLength > 0, "number of inputs must be bigger than 0"),
      weights = List.generate(
        inputLength,
        (_) => Value(Const.rand.nextDouble() * 2 - 1),
        growable: false,
      ),
      bias = Value(Const.rand.nextDouble() * 2 - 1);

  Value call(List<Value> inputs) {
    assert(
      inputs.length == weights.length,
      "inputs must be equal to neuron inputs",
    );

    Value result = weights[0] * inputs[0];
    for (int i = 1; i < weights.length; i++) {
      result = result + (weights[i] * inputs[i]);
    }
    result = result + bias;

    return result.tanh();
  }

  List<Value> get parameters => [...weights, bias];
}

class Layer {
  final List<Neuron> neurons;

  Layer({required int inputLength, required int outputLength})
    : assert(inputLength > 0, "number of inputs must be bigger than 0"),
      neurons = List.generate(
        outputLength,
        (_) => Neuron(inputLength: inputLength),
        growable: false,
      );

  List<Value> call(List<Value> inputs) {
    final result = <Value>[];

    for (final neuron in neurons) {
      result.add(neuron(inputs));
    }

    return result;
  }

  List<Value> get parameters {
    final result = <Value>[];
    for (final neuron in neurons) {
      result.addAll(neuron.parameters);
    }
    return result;
  }
}

class MLP {
  late final List<Layer> layers;

  MLP({required int inputLength, required List<int> outputLengths})
    : assert(inputLength > 0, "number of inputs must be bigger than 0"),
      assert(outputLengths.isNotEmpty, "output lengths must not be empty") {
    final combined = [inputLength, ...outputLengths];
    layers = List.generate(
      outputLengths.length,
      (i) => Layer(inputLength: combined[i], outputLength: combined[i + 1]),
    );
  }

  List<Value> call(List<Value> inputs) {
    for (final layer in layers) {
      inputs = layer(inputs);
    }
    return inputs;
  }

  List<Value> get parameters {
    final result = <Value>[];
    for (final layer in layers) {
      result.addAll(layer.parameters);
    }
    return result;
  }
}
