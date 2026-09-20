import 'dart:math' as math;

abstract final class Activations {
  static double tanh(double x) {
    if (x >= 0) {
      final exp = math.exp(-2 * x);
      return (1 - exp) / (1 + exp);
    }

    final exp = math.exp(2 * x);
    return (exp - 1) / (exp + 1);
  }

  static double dTanh(double x) {
    final t = tanh(x);
    return 1 - t * t;
  }

  static double sigmoid(double x) {
    if (x >= 0) {
      return 1 / (1 + math.exp(-x));
    }

    final exp = math.exp(x);
    return exp / (1 + exp);
  }

  static double dSigmoid(double x) {
    final s = sigmoid(x);
    return s * (1 - s);
  }

  static double relu(double x) {
    return x > 0 ? x : 0;
  }

  static double dRelu(double x) {
    return x > 0 ? 1 : 0;
  }

  static double leakyRelu(double x, {double alpha = 0.01}) {
    return x > 0 ? x : alpha * x;
  }

  static double dLeakyRelu(double x, {double alpha = 0.01}) {
    return x > 0 ? 1 : alpha;
  }

  static List<double> softmax(List<double> inputs) {
    if (inputs.isEmpty) {
      return [];
    }

    final maxValue = inputs.reduce(math.max);

    final exponentials = [
      for (final input in inputs) math.exp(input - maxValue),
    ];

    final sum = exponentials.reduce((left, right) => left + right);

    return [for (final exponential in exponentials) exponential / sum];
  }
}
