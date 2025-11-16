#!/usr/bin/env python3
"""
Diagnostic script to identify NNUE training issues
"""

import json
import numpy as np

def analyze_data_statistics(data_path='data/train.jsonl', max_samples=100000):
    """Analyze evaluation score statistics from training data"""
    if not data_path:
        print("No data file provided")
        return

    try:
        evaluations = []
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                item = json.loads(line)
                evaluations.append(float(item['eval']))

        eval_array = np.array(evaluations)

        print("=" * 80)
        print("TRAINING DATA STATISTICS")
        print("=" * 80)
        print(f"Samples analyzed: {len(evaluations):,}")
        print(f"Min evaluation: {eval_array.min():.2f}")
        print(f"Max evaluation: {eval_array.max():.2f}")
        print(f"Mean evaluation: {eval_array.mean():.2f}")
        print(f"Std deviation: {eval_array.std():.2f}")
        print(f"Median: {np.median(eval_array):.2f}")
        print(f"25th percentile: {np.percentile(eval_array, 25):.2f}")
        print(f"75th percentile: {np.percentile(eval_array, 75):.2f}")

        # After normalization
        normalized = (eval_array - eval_array.mean()) / eval_array.std()
        print("\n" + "=" * 80)
        print("AFTER Z-SCORE NORMALIZATION")
        print("=" * 80)
        print(f"Min normalized: {normalized.min():.2f}")
        print(f"Max normalized: {normalized.max():.2f}")
        print(f"Mean normalized: {normalized.mean():.2f}")
        print(f"Std normalized: {normalized.std():.2f}")

        # Check how many values exceed [-3, 3] (model output range)
        outside_range = np.sum((normalized < -3) | (normalized > 3))
        percent_outside = (outside_range / len(normalized)) * 100

        print("\n" + "=" * 80)
        print("MODEL OUTPUT RANGE ANALYSIS")
        print("=" * 80)
        print(f"Model output range (with tanh): [-3.0, 3.0]")
        print(f"Normalized values outside [-3, 3]: {outside_range:,} ({percent_outside:.2f}%)")

        if percent_outside > 1:
            print("\n⚠️  CRITICAL ISSUE FOUND:")
            print(f"   {percent_outside:.1f}% of normalized values exceed the model's output range!")
            print("   The tanh(x) * 3.0 output activation is too restrictive.")
            print("   The model cannot learn to predict values outside [-3, 3].")

        return eval_array, normalized

    except FileNotFoundError:
        print(f"Data file not found: {data_path}")
        print("Run training with auto_download=True to download data first")
        return None, None
    except Exception as e:
        print(f"Error analyzing data: {e}")
        return None, None


def identify_issues():
    """Identify potential issues in the NNUE model architecture"""
    print("\n" + "=" * 80)
    print("POTENTIAL ISSUES IN NNUE ARCHITECTURE")
    print("=" * 80)

    issues = []

    # Issue 1: Output scaling
    issues.append({
        'severity': 'CRITICAL',
        'issue': 'Output activation uses tanh(x) * 3.0',
        'problem': 'This limits output to [-3, 3], but normalized evaluations can exceed this range',
        'fix': 'Remove tanh activation or increase output_scale significantly'
    })

    # Issue 2: Weight initialization
    issues.append({
        'severity': 'MEDIUM',
        'issue': 'Xavier initialization uses gain=0.5',
        'problem': 'Small initial weights may slow down learning',
        'fix': 'Use gain=1.0 or try He initialization for ReLU-like activations'
    })

    # Issue 3: Small hidden layers
    issues.append({
        'severity': 'LOW',
        'issue': 'Network bottleneck: 768 -> 256 -> 32 -> 32 -> 1',
        'problem': 'Aggressive compression may lose information',
        'fix': 'Consider larger intermediate layers (e.g., 256 -> 64 -> 32 -> 1)'
    })

    for i, issue in enumerate(issues, 1):
        print(f"\n{i}. [{issue['severity']}] {issue['issue']}")
        print(f"   Problem: {issue['problem']}")
        print(f"   Fix: {issue['fix']}")


if __name__ == '__main__':
    import sys

    data_path = 'data/train.jsonl'
    if len(sys.argv) > 1:
        data_path = sys.argv[1]

    # Analyze data if available
    eval_array, normalized = analyze_data_statistics(data_path)

    # Always show architecture issues
    identify_issues()

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("1. IMMEDIATE: Remove tanh activation or use a larger output_scale (e.g., 10.0)")
    print("2. ALTERNATIVE: Remove output scaling entirely and let the model learn the range")
    print("3. Consider using linear output: x = self.fc3(x)  # No tanh")
    print("4. Increase weight initialization gain to 1.0")
    print("=" * 80)
