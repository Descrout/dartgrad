import 'dart:io';

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
    final probabilities = [1, 2, 3].valueList.softmax;

    expect(probabilities.reduce((a, b) => a + b), closeTo(1, 1e-12));
    expect(probabilities[0], lessThan(probabilities[1]));
    expect(probabilities[1], lessThan(probabilities[2]));
  });

  test('MLP produces the configured output and exposes its parameters', () {
    final model = MLP(inputLength: 3, outputLengths: [2, 1]);

    final output = model([1, 2, 3].valueList);

    expect(output, hasLength(1));
    expect(output.single.data, inInclusiveRange(-1, 1));
    expect(model.parameters, hasLength(11));
  });

  test('MLP saves and loads its architecture and parameters', () async {
    final directory = await Directory.systemTemp.createTemp('dartgrad_test_');
    final path = '${directory.path}${Platform.pathSeparator}model.json';

    try {
      final model = MLP(inputLength: 2, outputLengths: [3, 1]);
      for (var i = 0; i < model.parameters.length; i++) {
        model.parameters[i].data = i / 10;
      }
      final input = [0.5, -1.0].valueList;
      final expectedOutput = model(input).single.data;

      await model.save(path);
      final loaded = await MLP.load(path);

      expect(loaded.inputLength, 2);
      expect(loaded.outputLengths, [3, 1]);
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
