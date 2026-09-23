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
}
