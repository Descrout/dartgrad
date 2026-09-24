import 'dart:convert';
import 'dart:io';
import 'dart:math';

import 'value.dart';

final _random = Random();

List<Value> _initializeWeights(
  int inputLength,
  int outputLength,
  Activation activation,
) {
  final limit = switch (activation) {
    Activation.relu || Activation.leakyRelu => sqrt(6 / inputLength),
    _ => sqrt(6 / (inputLength + outputLength)),
  };

  return List.generate(
    inputLength,
    (_) => Value((_random.nextDouble() * 2 - 1) * limit),
    growable: false,
  );
}

/// Activation functions available to neurons and neural networks.
enum Activation {
  tanh,
  sigmoid,
  relu,
  leakyRelu,
  linear;

  /// Applies this activation function to [input].
  Value call(Value input) => switch (this) {
    tanh => input.tanh(),
    sigmoid => input.sigmoid(),
    relu => input.relu(),
    leakyRelu => input.leakyRelu(),
    linear => input,
  };
}

class Neuron {
  final List<Value> weights;
  final Value bias;
  final Activation activation;

  Neuron({required int inputLength, Activation activation = Activation.tanh})
    : this._(inputLength: inputLength, outputLength: 1, activation: activation);

  Neuron._({
    required int inputLength,
    required int outputLength,
    required this.activation,
  }) : assert(inputLength > 0, "number of inputs must be bigger than 0"),
       weights = _initializeWeights(inputLength, outputLength, activation),
       bias = Value(0);

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

    return activation(result);
  }

  List<Value> get parameters => [...weights, bias];
}

class Layer {
  final List<Neuron> neurons;
  final Activation activation;

  Layer({
    required int inputLength,
    required int outputLength,
    this.activation = Activation.tanh,
  }) : assert(inputLength > 0, "number of inputs must be bigger than 0"),
       assert(outputLength > 0, "number of outputs must be bigger than 0"),
       neurons = List.generate(
         outputLength,
         (_) => Neuron._(
           inputLength: inputLength,
           outputLength: outputLength,
           activation: activation,
         ),
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
  final int inputLength;
  final List<int> outputLengths;
  final List<Activation> activations;
  late final List<Layer> layers;

  MLP({
    required this.inputLength,
    required List<int> outputLengths,
    List<Activation>? activations,
  }) : assert(inputLength > 0, "number of inputs must be bigger than 0"),
       assert(outputLengths.isNotEmpty, "output lengths must not be empty"),
       assert(
         outputLengths.every((length) => length > 0),
         "output lengths must be bigger than 0",
       ),
       assert(
         activations == null || activations.length == outputLengths.length,
         "activations must match the number of layers",
       ),
       outputLengths = List.unmodifiable(outputLengths),
       activations = List.unmodifiable(
         activations ??
             List.filled(
               outputLengths.length,
               Activation.tanh,
               growable: false,
             ),
       ) {
    final combined = [inputLength, ...this.outputLengths];
    layers = List.generate(
      this.outputLengths.length,
      (i) => Layer(
        inputLength: combined[i],
        outputLength: combined[i + 1],
        activation: this.activations[i],
      ),
    );
  }

  /// Saves the model architecture, weights, and biases as JSON.
  Future<void> save(String path) async {
    final modelData = {
      'formatVersion': 2,
      'inputLength': inputLength,
      'outputLengths': outputLengths,
      'activations': activations.map((activation) => activation.name).toList(),
      'parameters': parameters.map((parameter) => parameter.data).toList(),
    };

    const encoder = JsonEncoder.withIndent('  ');
    await File(path).writeAsString(encoder.convert(modelData));
  }

  /// Loads a model architecture, weights, and biases from a JSON file.
  static Future<MLP> load(String path) async {
    final Object? decoded;
    try {
      decoded = jsonDecode(await File(path).readAsString());
    } on FormatException catch (error) {
      throw FormatException('Invalid dartgrad model JSON: ${error.message}');
    }

    if (decoded is! Map<String, dynamic>) {
      throw const FormatException('Model file must contain a JSON object.');
    }
    final formatVersion = decoded['formatVersion'];
    if (formatVersion != 1 && formatVersion != 2) {
      throw FormatException(
        'Unsupported model format version: $formatVersion.',
      );
    }

    final inputLength = decoded['inputLength'];
    final outputLengths = decoded['outputLengths'];
    final savedParameters = decoded['parameters'];

    if (inputLength is! int || inputLength <= 0) {
      throw const FormatException('inputLength must be a positive integer.');
    }
    if (outputLengths is! List ||
        outputLengths.isEmpty ||
        outputLengths.any((length) => length is! int || length <= 0)) {
      throw const FormatException(
        'outputLengths must contain positive integers.',
      );
    }
    if (savedParameters is! List ||
        savedParameters.any(
          (parameter) => parameter is! num || !parameter.isFinite,
        )) {
      throw const FormatException('parameters must contain finite numbers.');
    }

    final activations = formatVersion == 1
        ? List.filled(
            outputLengths.length,
            decoded['activation'] == null
                ? Activation.tanh
                : _parseActivation(decoded['activation']),
            growable: false,
          )
        : _parseActivations(decoded['activations'], outputLengths.length);

    final model = MLP(
      inputLength: inputLength,
      outputLengths: outputLengths.cast<int>(),
      activations: activations,
    );
    final modelParameters = model.parameters;
    if (savedParameters.length != modelParameters.length) {
      throw FormatException(
        'Expected ${modelParameters.length} parameters, '
        'but found ${savedParameters.length}.',
      );
    }

    for (var i = 0; i < modelParameters.length; i++) {
      modelParameters[i].data = (savedParameters[i] as num).toDouble();
    }

    return model;
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

Activation _parseActivation(Object? name) {
  for (final activation in Activation.values) {
    if (activation.name == name) return activation;
  }

  throw FormatException('Unsupported activation function: $name.');
}

List<Activation> _parseActivations(Object? names, int expectedLength) {
  if (names is! List || names.length != expectedLength) {
    throw FormatException(
      'activations must contain one value for each of the $expectedLength '
      'layers.',
    );
  }

  return names.map(_parseActivation).toList(growable: false);
}
