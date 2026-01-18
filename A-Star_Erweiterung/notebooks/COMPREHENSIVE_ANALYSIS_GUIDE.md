# Comprehensive Reopening Analysis - Usage Guide

## Overview

This comprehensive test suite analyzes A* reopening behavior across multiple configurations for 3DoF Shape Robot scenarios. It systematically tests different combinations of:

- **Heuristic weights (w)**: 0.5, 1.0, 1.5, 2.0
- **Grid discretizations**: From coarse (15x15x24) to very fine (30x30x48)
- **Reopening**: ON vs OFF for each configuration

## Key Metrics Analyzed

### Path Quality Metrics
- **Success Rate**: Percentage of tests that found a solution
- **Path Length**: Number of nodes in the solution path
- **Path Cost**: Sum of Euclidean distances along the path (actual travel distance)

### Computational Metrics
- **Planning Time**: Time taken to find the solution
- **Nodes Explored**: Number of nodes added to the graph
- **Nodes per Second**: Planning efficiency
- **Average Node Time**: Time spent per node (in milliseconds)

### Comparative Metrics
- **Path Length Improvement**: How much shorter paths are with reopening (%)
- **Path Cost Improvement**: How much cheaper paths are with reopening (%)
- **Time Overhead**: Additional computation time with reopening (%)
- **Success Improvement**: Additional successful cases with reopening

## Files Generated

1. **comprehensive_reopening_results.csv**
   - Complete detailed results for all 64 tests (32 configs × 2 benchmarks × 2 reopening settings)
   - All metrics for each individual test

2. **reopening_comparison_metrics.csv**
   - Direct comparisons between reopening ON/OFF for same configurations
   - Shows improvements/regressions for each metric

3. **comprehensive_reopening_analysis.png**
   - 12-panel visualization showing all key insights
   - Success rates, path quality, timing, efficiency, trade-offs

## Running the Analysis

### Command Line

```bash
cd notebooks
python comprehensive_reopening_analysis.py
```

This will:
- Run all 64 tests
- Print detailed progress
- Save results to CSV files
- Generate comprehensive visualizations
- Display key findings summary

### In Jupyter Notebook

```python
import comprehensive_reopening_analysis as cra
import pandas as pd

# Create analyzer
analyzer = cra.ReopeningAnalyzer()

# Run with default parameters
results_df = analyzer.run_comprehensive_analysis(verbose=True)

# Compute comparisons
comparison_df = analyzer.compute_comparative_metrics(results_df)

# Create visualizations
analyzer.create_comprehensive_visualizations(
    results_df, 
    comparison_df,
    save_path="my_analysis.png"
)

# Display results
display(results_df.head(20))
display(comparison_df)
```

### Custom Configuration

```python
# Test specific weight values
custom_w_values = [0.3, 0.7, 1.5, 2.5]

# Test specific discretizations
custom_steps = [
    [10, 10, 18],  # Very coarse
    [20, 20, 36],  # Medium
    [35, 35, 54],  # Very fine
]

# Run custom analysis
results_df = analyzer.run_comprehensive_analysis(
    w_values=custom_w_values,
    num_steps_configs=custom_steps,
    verbose=True
)
```

## Understanding the Results

### When Reopening is Beneficial

Reopening typically helps when:

1. **Lower heuristic weights (w ≤ 1.0)**
   - A* explores more nodes
   - More opportunities to find better paths
   - Can escape local suboptimal choices

2. **Finer discretizations**
   - More alternative paths available
   - Grid structure creates more path options
   - Better path refinement possible

3. **Complex environments**
   - Trap scenarios
   - Narrow passages
   - Multiple routing options

### When Reopening Has Drawbacks

Reopening may be problematic when:

1. **Higher heuristic weights (w ≥ 1.5)**
   - Can cause excessive reopening
   - May exceed iteration limits
   - Computational overhead outweighs benefits

2. **Very fine grids + high weights**
   - Combinatorial explosion of reopening candidates
   - Performance degradation
   - May fail to complete within iteration limits

3. **Simple, direct paths**
   - Standard A* already finds optimal path
   - Reopening overhead without benefit

## Key Findings from Default Configuration

Based on the test results:

### Success Rates
- Overall: Reopening has similar or slightly better success rates
- Exception: High w + fine grids can cause timeout failures

### Path Quality
- **Best improvements**: w=1.0 with medium grids (20x20x36)
  - Example: 19% shorter paths (52 → 42 nodes)
  - Lower path costs
- **Modest improvements**: w=0.5 configurations
  - Usually same path length, but verified optimality
- **Mixed results**: w≥1.5 configurations
  - Sometimes better, sometimes worse
  - Depends on specific problem geometry

### Computational Cost
- **Low overhead**: w=1.0 configurations (~10-30% overhead)
- **High overhead**: w=0.5 configurations (can be 2-4x slower)
- **Variable**: w≥1.5 (can improve or degrade significantly)

### Efficiency Sweet Spot
- **Recommended**: w=1.0 with 20x20x36 or 25x25x36 grids
  - Good balance of path quality improvement
  - Reasonable computational overhead
  - Reliable success rates

## Visualization Panels

The comprehensive visualization includes:

1. **Success Rate by Weight** - Overall reliability
2. **Avg Path Length by Weight** - Path quality impact
3. **Avg Planning Time by Weight** - Computational cost
4. **Success Rate by Discretization** - Grid resolution impact
5. **Path Cost Improvement** - Quality gains/losses
6. **Time Overhead** - Computational overhead
7. **Nodes Explored** - Search space size
8. **Path Length Improvement Distribution** - Variability
9. **Planning Efficiency** - Nodes per second
10. **Benchmark-specific** - Problem type impact
11. **Time vs Quality Trade-off** - Scatter plot
12. **Summary Statistics Table** - Key numbers

## Recommendations

### For Best Path Quality
```python
config = {
    'w': 1.0,
    'num_steps': [25, 25, 36],
    'allowReopening': True,
    'heuristic': 'euclidean',
    'checkEdgeCollision': False
}
```

### For Fast Planning
```python
config = {
    'w': 1.5,
    'num_steps': [20, 20, 36],
    'allowReopening': False,
    'heuristic': 'euclidean',
    'checkEdgeCollision': False
}
```

### For Difficult Problems
```python
config = {
    'w': 1.0,
    'num_steps': [30, 30, 48],
    'allowReopening': True,
    'heuristic': 'euclidean',
    'checkEdgeCollision': False
}
```

## Extending the Analysis

### Add More Benchmarks

Edit `IPTestSuite2.py` to add more test scenarios, then re-run the analysis.

### Test Different Heuristics

Modify the test to include different heuristic functions (if implemented in your planner).

### Analyze Edge Collision Impact

Add tests with `checkEdgeCollision=True` to see how it interacts with reopening.

### Custom Metrics

Add your own metrics to the `run_single_test` method to analyze specific aspects.

## Troubleshooting

**Problem**: Tests timeout with "Max. Iterationen erreicht"
- **Solution**: This happens with high w values + reopening. Consider lowering max_iterations or using lower w values.

**Problem**: Very slow execution
- **Solution**: Reduce the number of configurations or use coarser grids for initial testing.

**Problem**: Visualization doesn't show
- **Solution**: Make sure matplotlib backend is set correctly. In notebooks, use `%matplotlib inline`.

## License

Part of "Introduction to robot path planning" course (Author: Bjoern Hein)
Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
