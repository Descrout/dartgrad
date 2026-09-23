import 'dart:math' as math;

import 'activations.dart';
import 'value.dart';

enum Ops {
  add("+"),
  mult("*"),
  tanh("tanh"),
  sigmoid("sigmoid"),
  relu("relu"),
  leakyRelu("leakyRelu"),
  exp("exp"),
  pow("pow");

  const Ops(this.label);
  final String label;

  @override
  String toString() => label;
}

class Operation {
  Operation({required this.left, this.right, required this.op});
  Value left;
  Value? right;

  void backward(Value from) {
    final localDerivative = switch (op) {
      .add => (1, 1),
      .mult => (right!.data, left.data),
      .tanh => (Activations.dTanh(left.data), 0),
      .sigmoid => (Activations.dSigmoid(left.data), 0),
      .relu => (Activations.dRelu(left.data), 0),
      .leakyRelu => (Activations.dLeakyRelu(left.data), 0),
      .exp => (math.exp(left.data), 0),
      .pow => (right!.data * math.pow(left.data, right!.data - 1), 0),
    };

    left.grad += from.grad * localDerivative.$1;
    if (right != null) {
      right!.grad += from.grad * localDerivative.$2;
    }
  }

  Ops op;
}
