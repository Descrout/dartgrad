import 'dart:math' as math;

import 'package:dartgrad/dartgrad.dart';

final _random = math.Random();

extension NumListX on List<num> {
  List<Value> get valueList =>
      map((e) => Value(e.toDouble())).toList(growable: false);
}

extension ValueListX on List<Value> {
  /// Returns differentiable, normalized probabilities for these logits.
  List<Value> get softmax {
    if (isEmpty) return [];

    final maxValue = reduce(
      (current, next) => next.data > current.data ? next : current,
    );

    final exponentials = map(
      (value) => (value - maxValue).exp(),
    ).toList(growable: false);

    final sum = exponentials.reduce((total, value) => total + value);

    return exponentials.map((value) => value / sum).toList(growable: false);
  }

  /// Calculates numerically stable cross-entropy loss for a class index.
  Value crossEntropy({required int targetClass}) {
    if (isEmpty) {
      throw StateError('Cannot calculate cross-entropy for an empty list.');
    }
    RangeError.checkValidIndex(targetClass, this, 'targetClass');

    final maxValue = reduce(
      (current, next) => next.data > current.data ? next : current,
    );
    final exponentialSum = map(
      (value) => (value - maxValue).exp(),
    ).reduce((total, value) => total + value);

    return exponentialSum.log() + maxValue - this[targetClass];
  }

  int get argmax {
    assert(isNotEmpty, 'Cannot pick from an empty list.');

    double max = double.negativeInfinity;
    int idx = -1;

    for (int i = 0; i < length; i++) {
      if (this[i].data > max) {
        idx = i;
        max = this[i].data;
      }
    }

    return idx;
  }

  int get argmin {
    assert(isNotEmpty, 'Cannot pick from an empty list.');

    double min = double.infinity;
    int idx = -1;

    for (int i = 0; i < length; i++) {
      if (this[i].data < min) {
        idx = i;
        min = this[i].data;
      }
    }

    return idx;
  }

  int get weightedRandPick {
    assert(isNotEmpty, 'Cannot pick from an empty list.');

    double total = 0.0;

    for (final weight in this) {
      if (!weight.data.isFinite || weight.data < 0) {
        throw ArgumentError('weights must be finite and non-negative.');
      }

      total += weight.data;
    }

    if (total <= 0) {
      throw StateError('At least one weight must be greater than zero.');
    }

    final target = _random.nextDouble() * total;
    var cumulative = 0.0;

    for (var i = 0; i < length; i++) {
      cumulative += this[i].data;

      if (target < cumulative) {
        return i;
      }
    }

    return lastIndexWhere((weight) => weight.data > 0);
  }
}

class Value {
  Value(this.data, {this.prev, this.label});

  double data;
  Operation? prev;

  String? label;

  double grad = 0;

  void backward() {
    final topo = <Value>[];
    final visited = <Value>{};

    void buildTopo(Value value) {
      if (!visited.add(value)) return;

      final operation = value.prev;
      if (operation != null) {
        buildTopo(operation.left);
        final right = operation.right;
        if (right != null) buildTopo(right);
      }
      topo.add(value);
    }

    buildTopo(this);
    for (final value in topo) {
      value.grad = 0;
    }

    grad = 1;
    for (final value in topo.reversed) {
      value.prev?.backward(value);
    }
  }

  Value operator +(Object other) {
    final o = other is num ? Value(other.toDouble()) : other as Value;

    return Value(
      data + o.data,
      prev: Operation(left: this, right: o, op: .add),
    );
  }

  Value operator -() {
    return this * -1;
  }

  Value operator -(Object other) {
    final o = other is num ? Value(other.toDouble()) : other as Value;
    return this + (-o);
  }

  Value operator *(Object other) {
    final o = other is num ? Value(other.toDouble()) : other as Value;

    return Value(
      data * o.data,
      prev: Operation(left: this, right: o, op: .mult),
    );
  }

  Value operator /(Object other) {
    final o = other is num ? Value(other.toDouble()) : other as Value;
    return this * (o.pow(-1));
  }

  Value tanh() {
    return Value(
      Activations.tanh(data),
      prev: Operation(left: this, op: .tanh),
    );
  }

  Value sigmoid() {
    return Value(
      Activations.sigmoid(data),
      prev: Operation(left: this, op: .sigmoid),
    );
  }

  Value relu() {
    return Value(
      Activations.relu(data),
      prev: Operation(left: this, op: .relu),
    );
  }

  Value leakyRelu() {
    return Value(
      Activations.leakyRelu(data),
      prev: Operation(left: this, op: .leakyRelu),
    );
  }

  Value exp() {
    return Value(
      math.exp(data),
      prev: Operation(left: this, op: .exp),
    );
  }

  Value log() {
    return Value(
      math.log(data),
      prev: Operation(left: this, op: .log),
    );
  }

  Value pow(double value) {
    return Value(
      math.pow(data, value).toDouble(),
      prev: Operation(left: this, right: Value(value), op: .pow),
    );
  }

  @override
  String toString() => "Value($data, grad: $grad)";

  String diagram() =>
      _render().lines.map((l) => l.replaceAll(RegExp(r'\s+$'), '')).join('\n');

  void show() {
    print(diagram());
  }

  static String _num(double v) => v.toStringAsFixed(4);

  String get _box {
    final l = label == null ? '' : '$label | ';
    return '[ ${l}data ${_num(data)} | grad ${_num(grad)} ]';
  }

  _Block _render() {
    final p = prev;
    if (p == null) return _Block([_box], 0);

    final l = p.left._render();
    final right = p.right;
    if (right == null) {
      final out = <String>[];
      for (var i = 0; i < l.lines.length; i++) {
        final child = _pad(l.lines[i], l.width, i == l.anchor);
        out.add(i == l.anchor ? '$child── ( ${p.op} ) ─ $_box' : child);
      }
      return _Block(out, l.anchor);
    }

    final r = right._render();
    final w = l.width > r.width ? l.width : r.width;

    final child = <String>[];
    for (var i = 0; i < l.lines.length; i++) {
      child.add(_pad(l.lines[i], w, i == l.anchor));
    }
    child.add(_pad('', w, false));
    for (var i = 0; i < r.lines.length; i++) {
      child.add(_pad(r.lines[i], w, i == r.anchor));
    }

    final la = l.anchor;
    final ra = l.lines.length + 1 + r.anchor;
    final pa = (la + ra) ~/ 2;

    final out = <String>[];
    for (var i = 0; i < child.length; i++) {
      if (i == pa) {
        out.add('${child[i]}  ├── ( ${p.op} ) ─ $_box');
      } else if (i == la) {
        out.add('${child[i]}──┐');
      } else if (i == ra) {
        out.add('${child[i]}──┘');
      } else if (i > la && i < ra) {
        out.add('${child[i]}  │');
      } else {
        out.add('${child[i]}   ');
      }
    }

    return _Block(out, pa);
  }

  static String _pad(String s, int w, bool isAnchor) =>
      s + (isAnchor ? '─' : ' ') * (w - s.length);
}

class _Block {
  _Block(this.lines, this.anchor);

  final List<String> lines;
  final int anchor;

  int get width => lines.fold(0, (m, l) => l.length > m ? l.length : m);
}
