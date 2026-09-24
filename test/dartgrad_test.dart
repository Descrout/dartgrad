import 'dart:io';
import 'dart:math' as math;

import 'package:dartgrad/dartgrad.dart';
import 'package:test/test.dart';

void main() {
  group('Value', () {
    test('evaluates an expression and calculates gradients', () {
      final x = Value(2);
      final result = (x * 3 + 2).pow(2);

      result.backward();

      expect(result.data, 64);
      expect(x.grad, 48);
    });

    test('applies activation functions', () {
      expect(Value(-2).relu().data, 0);
      expect(Value(-2).leakyRelu().data, closeTo(-0.02, 1e-12));
      expect(Value(0).sigmoid().data, 0.5);
      expect(Value(0).tanh().data, 0);
    });
  });

  test('softmax returns normalized probabilities', () {
    final logits = [1, 2, 3].valueList;
    final probabilities = logits.softmax;

    expect(
      probabilities
          .map((probability) => probability.data)
          .reduce((a, b) => a + b),
      closeTo(1, 1e-12),
    );
    expect(probabilities[0].data, lessThan(probabilities[1].data));
    expect(probabilities[1].data, lessThan(probabilities[2].data));

    probabilities.last.backward();
    expect(logits.last.grad, greaterThan(0));
    expect(logits.first.grad, lessThan(0));
  });

  test('cross-entropy calculates loss and gradients from logits', () {
    final logits = [1, 2, 3].valueList;
    final probabilities = logits.softmax;
    final loss = logits.crossEntropy(targetClass: 2);

    loss.backward();

    expect(loss.data, closeTo(0.4076059644, 1e-10));
    for (var i = 0; i < logits.length; i++) {
      final expectedGradient = probabilities[i].data - (i == 2 ? 1 : 0);
      expect(logits[i].grad, closeTo(expectedGradient, 1e-12));
    }
  });

  test('MLP produces the configured output and exposes its parameters', () {
    final model = MLP(inputLength: 3, outputLengths: [2, 1]);

    final output = model([1, 2, 3].valueList);

    expect(output, hasLength(1));
    expect(output.single.data, inInclusiveRange(-1, 1));
    expect(model.parameters, hasLength(11));
    expect(model.activations, [Activation.tanh, Activation.tanh]);
  });

  test('Neuron uses the selected activation function', () {
    final neuron = Neuron(inputLength: 1, activation: Activation.relu);
    neuron.weights.single.data = 1;
    neuron.bias.data = 0;

    expect(neuron([-2].valueList).data, 0);
    expect(neuron([2].valueList).data, 2);
  });

  test('Layer uses activation-aware weight initialization', () {
    final xavierLayer = Layer(
      inputLength: 4,
      outputLength: 3,
      activation: Activation.tanh,
    );
    final heLayer = Layer(
      inputLength: 4,
      outputLength: 3,
      activation: Activation.relu,
    );
    final xavierLimit = math.sqrt(6 / 7);
    final heLimit = math.sqrt(6 / 4);

    expect(
      xavierLayer.neurons.expand((neuron) => neuron.weights),
      everyElement(
        predicate<Value>((weight) => weight.data.abs() <= xavierLimit),
      ),
    );
    expect(
      heLayer.neurons.expand((neuron) => neuron.weights),
      everyElement(predicate<Value>((weight) => weight.data.abs() <= heLimit)),
    );
    expect(
      [
        ...xavierLayer.neurons,
        ...heLayer.neurons,
      ].map((neuron) => neuron.bias.data),
      everyElement(0),
    );
  });

  test('MLP applies a separate activation to each layer', () {
    final model = MLP(
      inputLength: 1,
      outputLengths: [1, 1],
      activations: [Activation.relu, Activation.sigmoid],
    );
    for (final layer in model.layers) {
      layer.neurons.single.weights.single.data = 1;
      layer.neurons.single.bias.data = 0;
    }

    expect(model([-2].valueList).single.data, 0.5);
    expect(
      () => MLP(
        inputLength: 1,
        outputLengths: [1, 1],
        activations: [Activation.relu],
      ),
      throwsA(isA<AssertionError>()),
    );
  });

  test('MLP saves and loads its architecture and parameters', () async {
    final directory = await Directory.systemTemp.createTemp('dartgrad_test_');
    final path = '${directory.path}${Platform.pathSeparator}model.json';

    try {
      final model = MLP(
        inputLength: 2,
        outputLengths: [3, 1],
        activations: [Activation.relu, Activation.sigmoid],
      );
      for (var i = 0; i < model.parameters.length; i++) {
        model.parameters[i].data = i / 10;
      }
      final input = [0.5, -1.0].valueList;
      final expectedOutput = model(input).single.data;

      await model.save(path);
      final loaded = await MLP.load(path);

      expect(loaded.inputLength, 2);
      expect(loaded.outputLengths, [3, 1]);
      expect(loaded.activations, [Activation.relu, Activation.sigmoid]);
      expect(
        loaded.parameters.map((parameter) => parameter.data),
        model.parameters.map((parameter) => parameter.data),
      );
      expect(loaded(input).single.data, closeTo(expectedOutput, 1e-12));
    } finally {
      await directory.delete(recursive: true);
    }
  });
}
