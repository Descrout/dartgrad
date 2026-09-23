import 'dart:convert';
import 'dart:io';
import 'dart:math';

import 'value.dart';

final _random = Random();

class Neuron {
  final List<Value> weights;
  final Value bias;

  Neuron({required int inputLength})
    : assert(inputLength > 0, "number of inputs must be bigger than 0"),
      weights = List.generate(
        inputLength,
        (_) => Value(_random.nextDouble() * 2 - 1),
        growable: false,
      ),
      bias = Value(_random.nextDouble() * 2 - 1);

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
  final int inputLength;
  final List<int> outputLengths;
  late final List<Layer> layers;

  MLP({required this.inputLength, required List<int> outputLengths})
    : assert(inputLength > 0, "number of inputs must be bigger than 0"),
      assert(outputLengths.isNotEmpty, "output lengths must not be empty"),
      outputLengths = List.unmodifiable(outputLengths) {
    final combined = [inputLength, ...this.outputLengths];
    layers = List.generate(
      this.outputLengths.length,
      (i) => Layer(inputLength: combined[i], outputLength: combined[i + 1]),
    );
  }

  /// Saves the model architecture, weights, and biases as JSON.
  Future<void> save(String path) async {
    final modelData = {
      'formatVersion': 1,
      'inputLength': inputLength,
      'outputLengths': outputLengths,
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
    if (decoded['formatVersion'] != 1) {
      throw FormatException(
        'Unsupported model format version: ${decoded['formatVersion']}.',
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

    final model = MLP(
      inputLength: inputLength,
      outputLengths: outputLengths.cast<int>(),
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
