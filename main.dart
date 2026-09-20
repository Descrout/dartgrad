import 'mlp.dart';
import 'value.dart';

void main() {
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

  // Forward pass
  for (int i = 0; i < 50; i++) {
    ypred = [for (final x in xs) n(x.valueList).softmaxPick];
    loss = [for (int i = 0; i < ypred.length; i++) (ypred[i] - ys[i]).pow(2)]
        .reduce((a, b) => a + b);

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
