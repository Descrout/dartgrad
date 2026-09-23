import 'package:dartgrad/dartgrad.dart';

void main() {
  singleOutput();
  multipleOutput();
}

void singleOutput() {
  final xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
  ];

  final ys = [1.0, -1.0, -1.0, 1.0].valueList;

  final n = MLP(inputLength: 3, outputLengths: [4, 4, 1]);

  late List<Value> ypred;
  late Value loss;

  for (int i = 0; i < 100; i++) {
    // Forward pass
    ypred = [for (final x in xs) n(x.valueList)[0]];
    loss =
        [
          for (int i = 0; i < ypred.length; i++) (ypred[i] - ys[i]).pow(2),
        ].reduce((a, b) => a + b) /
        ypred.length;

    // Backward pass
    loss.backward();

    // Gradient descent
    for (final p in n.parameters) {
      p.data += -0.1 * p.grad;
    }

    print("Step: ${i + 1}, Loss: ${loss.data}");
  }

  print(ypred.map((e) => e.data).toList());
}

void multipleOutput() {
  final xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
  ];
  final ys = [
    [1.0, -1.0],
    [-1.0, 1.0],
    [-1.0, 1.0],
    [1.0, -1.0],
  ].map((y) => y.valueList).toList();

  final n = MLP(inputLength: 3, outputLengths: [4, 4, 2]);

  late List<List<Value>> ypred;
  late Value loss;

  for (int step = 0; step < 1000; step++) {
    ypred = [for (final x in xs) n(x.valueList)];

    final errors = <Value>[];

    for (int sample = 0; sample < ypred.length; sample++) {
      for (int output = 0; output < 2; output++) {
        errors.add((ypred[sample][output] - ys[sample][output]).pow(2));
      }
    }

    loss = errors.reduce((total, error) => total + error);
    loss = loss / errors.length;

    loss.backward();

    for (final p in n.parameters) {
      p.data += -0.1 * p.grad;
    }

    print('Step: ${step + 1}, Loss: ${loss.data}');
  }

  print(ypred.map((e) => e.map((e) => e.data).toList()).toList());
}
