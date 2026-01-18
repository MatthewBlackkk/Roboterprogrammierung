#!/usr/bin/env python
"""
Comprehensive Reopening Analysis for 3DoF Shape Robot
Tests reopening across multiple configurations to identify:
- When reopening provides benefits (shorter paths, better solutions)
- When reopening has drawbacks (computation overhead)
- Optimal configuration combinations
"""

import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import AStarReopeningNew
import IPTestSuite2 as ts


class ReopeningAnalyzer:
    """Comprehensive analyzer for A* reopening behavior"""

    def __init__(self):
        self.results = []

    def run_single_test(self, benchmark, config: Dict, test_id: str) -> Dict:
        """
        Run a single test configuration

        Returns detailed metrics including:
        - Success/failure
        - Path quality metrics
        - Computational metrics
        - Graph statistics
        """
        result = {
            'test_id': test_id,
            'benchmark_name': benchmark.name,
            'w': config['w'],
            'num_steps_x': config['num_steps'][0],
            'num_steps_y': config['num_steps'][1],
            'num_steps_theta': config['num_steps'][2],
            'total_grid_points': config['num_steps'][0] * config['num_steps'][1] * config['num_steps'][2],
            'allowReopening': config['allowReopening'],
            'checkEdgeCollision': config.get('checkEdgeCollision', False),
            'success': False,
            'path_length': 0,
            'path_cost': 0.0,
            'num_nodes_explored': 0,
            'num_nodes_in_graph': 0,
            'planning_time': 0.0,
            'nodes_per_second': 0.0,
            'avg_node_time_ms': 0.0,
            'graph_density': 0.0,
            'error': None
        }

        try:
            planner = AStarReopeningNew.ReopenAStar(benchmark.collisionChecker)

            start_time = time.time()
            solution = planner.planPath(benchmark.startList, benchmark.goalList, config)
            end_time = time.time()

            planning_time = end_time - start_time
            result['planning_time'] = planning_time

            if solution:
                result['success'] = True
                result['path_length'] = len(solution)

                # Calculate path cost (sum of euclidean distances)
                path_cost = 0.0
                for i in range(len(solution) - 1):
                    node1 = planner.graph.nodes[solution[i]]
                    node2 = planner.graph.nodes[solution[i+1]]
                    path_cost += np.linalg.norm(np.array(node1['pos']) - np.array(node2['pos']))
                result['path_cost'] = path_cost

                # Graph statistics
                num_nodes = planner.graph.number_of_nodes()
                num_edges = planner.graph.number_of_edges()
                result['num_nodes_in_graph'] = num_nodes
                result['num_nodes_explored'] = num_nodes  # In A*, all nodes in graph were explored

                if planning_time > 0:
                    result['nodes_per_second'] = num_nodes / planning_time
                    result['avg_node_time_ms'] = (planning_time / num_nodes) * 1000

                if num_nodes > 1:
                    max_edges = num_nodes * (num_nodes - 1)
                    result['graph_density'] = num_edges / max_edges if max_edges > 0 else 0
            else:
                result['error'] = "No path found"
                result['num_nodes_in_graph'] = planner.graph.number_of_nodes()

        except Exception as e:
            result['error'] = str(e)
            result['planning_time'] = time.time() - start_time

        return result

    def run_comprehensive_analysis(
        self,
        benchmarks: List = None,
        w_values: List[float] = None,
        num_steps_configs: List[List[int]] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Run comprehensive analysis across multiple configurations

        Args:
            benchmarks: List of benchmark scenarios (default: from IPTestSuite2)
            w_values: List of heuristic weights to test
            num_steps_configs: List of discretization configurations [x, y, theta]
            verbose: Print progress

        Returns:
            DataFrame with all test results
        """
        if benchmarks is None:
            benchmarks = ts.benchList

        if w_values is None:
            # Test different heuristic weights
            w_values = [0.5, 1.0, 1.5, 2.0]

        if num_steps_configs is None:
            # Test different discretization levels
            num_steps_configs = [
                [15, 15, 24],  # Coarse - fast but less accurate
                [20, 20, 36],  # Medium - balanced
                [25, 25, 36],  # Fine - more accurate but slower
                [30, 30, 48],  # Very fine - highest accuracy
            ]

        self.results = []
        total_tests = len(benchmarks) * len(w_values) * len(num_steps_configs) * 2  # x2 for with/without reopening
        test_count = 0

        if verbose:
            print("="*80)
            print("COMPREHENSIVE REOPENING ANALYSIS FOR 3DOF SHAPE ROBOT")
            print("="*80)
            print(f"Benchmarks: {len(benchmarks)}")
            print(f"Weight values (w): {w_values}")
            print(f"Discretization configs: {len(num_steps_configs)}")
            print(f"Total tests: {total_tests}")
            print("="*80)

        for bench_idx, benchmark in enumerate(benchmarks):
            if verbose:
                print(f"\n[Benchmark {bench_idx+1}/{len(benchmarks)}] {benchmark.name}")
                print(f"  Description: {benchmark.description}")
                print(f"  Difficulty: Level {benchmark.level}")

            for w in w_values:
                for steps in num_steps_configs:
                    # Test WITHOUT reopening
                    test_count += 1
                    config_no_reopen = {
                        'heuristic': 'euclidean',
                        'w': w,
                        'num_steps': steps,
                        'checkEdgeCollision': False,
                        'allowReopening': False
                    }

                    test_id = f"B{bench_idx+1}_w{w}_s{steps[0]}x{steps[1]}x{steps[2]}_NoReopen"
                    if verbose:
                        print(f"  [{test_count}/{total_tests}] w={w}, steps={steps}, reopening=OFF...", end=" ")

                    result_no = self.run_single_test(benchmark, config_no_reopen, test_id)
                    self.results.append(result_no)

                    if verbose:
                        status = "[OK]" if result_no['success'] else "[FAIL]"
                        print(f"{status} t={result_no['planning_time']:.3f}s, "
                              f"path={result_no['path_length']}, nodes={result_no['num_nodes_in_graph']}")

                    # Test WITH reopening
                    test_count += 1
                    config_with_reopen = config_no_reopen.copy()
                    config_with_reopen['allowReopening'] = True

                    test_id = f"B{bench_idx+1}_w{w}_s{steps[0]}x{steps[1]}x{steps[2]}_WithReopen"
                    if verbose:
                        print(f"  [{test_count}/{total_tests}] w={w}, steps={steps}, reopening=ON...", end=" ")

                    result_with = self.run_single_test(benchmark, config_with_reopen, test_id)
                    self.results.append(result_with)

                    if verbose:
                        status = "[OK]" if result_with['success'] else "[FAIL]"
                        print(f"{status} t={result_with['planning_time']:.3f}s, "
                              f"path={result_with['path_length']}, nodes={result_with['num_nodes_in_graph']}")

        if verbose:
            print("\n" + "="*80)
            print("ANALYSIS COMPLETE")
            print("="*80)

        df = pd.DataFrame(self.results)
        return df

    def compute_comparative_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute comparative metrics between reopening ON/OFF for same configurations

        Returns DataFrame showing improvements/regressions
        """
        comparisons = []

        # Group by configuration (same benchmark, w, num_steps)
        grouped = df.groupby(['benchmark_name', 'w', 'num_steps_x', 'num_steps_y', 'num_steps_theta'])

        for name, group in grouped:
            if len(group) != 2:
                continue

            no_reopen = group[group['allowReopening'] == False].iloc[0]
            with_reopen = group[group['allowReopening'] == True].iloc[0]

            comparison = {
                'benchmark_name': name[0],
                'w': name[1],
                'num_steps': f"{name[2]}x{name[3]}x{name[4]}",
                'grid_size': no_reopen['total_grid_points'],

                # Success comparison
                'success_without': no_reopen['success'],
                'success_with': with_reopen['success'],
                'success_improvement': int(with_reopen['success']) - int(no_reopen['success']),

                # Path length comparison
                'path_length_without': no_reopen['path_length'],
                'path_length_with': with_reopen['path_length'],
                'path_length_diff': with_reopen['path_length'] - no_reopen['path_length'],
                'path_length_improvement_pct': self._calc_improvement_pct(
                    no_reopen['path_length'], with_reopen['path_length']
                ),

                # Path cost comparison
                'path_cost_without': no_reopen['path_cost'],
                'path_cost_with': with_reopen['path_cost'],
                'path_cost_improvement_pct': self._calc_improvement_pct(
                    no_reopen['path_cost'], with_reopen['path_cost']
                ),

                # Time comparison
                'time_without': no_reopen['planning_time'],
                'time_with': with_reopen['planning_time'],
                'time_overhead_pct': self._calc_overhead_pct(
                    no_reopen['planning_time'], with_reopen['planning_time']
                ),

                # Node exploration comparison
                'nodes_without': no_reopen['num_nodes_in_graph'],
                'nodes_with': with_reopen['num_nodes_in_graph'],
                'nodes_diff': with_reopen['num_nodes_in_graph'] - no_reopen['num_nodes_in_graph'],
            }

            comparisons.append(comparison)

        return pd.DataFrame(comparisons)

    def _calc_improvement_pct(self, baseline, new_value):
        """Calculate improvement percentage (negative = better for minimization)"""
        if baseline == 0 or not baseline:
            return 0
        return ((new_value - baseline) / baseline) * 100

    def _calc_overhead_pct(self, baseline, new_value):
        """Calculate overhead percentage (positive = more overhead)"""
        if baseline == 0:
            return 0
        return ((new_value - baseline) / baseline) * 100

    def analyze_by_weight(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze how different weights affect reopening benefit"""
        summary = df.groupby(['w', 'allowReopening']).agg({
            'success': 'mean',
            'path_length': 'mean',
            'path_cost': 'mean',
            'planning_time': 'mean',
            'num_nodes_in_graph': 'mean'
        }).round(3)

        return summary

    def analyze_by_discretization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze how grid discretization affects reopening benefit"""
        df['discretization'] = df['num_steps_x'].astype(str) + 'x' + \
                               df['num_steps_y'].astype(str) + 'x' + \
                               df['num_steps_theta'].astype(str)

        summary = df.groupby(['discretization', 'allowReopening']).agg({
            'success': 'mean',
            'path_length': 'mean',
            'path_cost': 'mean',
            'planning_time': 'mean',
            'num_nodes_in_graph': 'mean'
        }).round(3)

        return summary

    def create_comprehensive_visualizations(self, df: pd.DataFrame, comparison_df: pd.DataFrame, save_path: str = None):
        """Create comprehensive visualization suite"""

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'

        # Create large figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

        # 1. Success Rate by Weight
        ax1 = fig.add_subplot(gs[0, 0])
        success_by_w = df.groupby(['w', 'allowReopening'])['success'].mean().unstack()
        success_by_w.plot(kind='bar', ax=ax1, color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black')
        ax1.set_title('Success Rate by Weight (w)', fontweight='bold', fontsize=11)
        ax1.set_ylabel('Success Rate')
        ax1.set_xlabel('Weight (w)')
        ax1.legend(['Without Reopening', 'With Reopening'], loc='lower right')
        ax1.set_ylim([0, 1.1])
        ax1.grid(axis='y', alpha=0.3)

        # 2. Average Path Length by Weight
        ax2 = fig.add_subplot(gs[0, 1])
        success_df = df[df['success'] == True]
        if len(success_df) > 0:
            path_by_w = success_df.groupby(['w', 'allowReopening'])['path_length'].mean().unstack()
            path_by_w.plot(kind='bar', ax=ax2, color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black')
            ax2.set_title('Avg Path Length by Weight', fontweight='bold', fontsize=11)
            ax2.set_ylabel('Path Length (nodes)')
            ax2.set_xlabel('Weight (w)')
            ax2.legend(['Without Reopening', 'With Reopening'])
            ax2.grid(axis='y', alpha=0.3)

        # 3. Planning Time by Weight
        ax3 = fig.add_subplot(gs[0, 2])
        time_by_w = df.groupby(['w', 'allowReopening'])['planning_time'].mean().unstack()
        time_by_w.plot(kind='bar', ax=ax3, color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black')
        ax3.set_title('Avg Planning Time by Weight', fontweight='bold', fontsize=11)
        ax3.set_ylabel('Time (seconds)')
        ax3.set_xlabel('Weight (w)')
        ax3.legend(['Without Reopening', 'With Reopening'])
        ax3.grid(axis='y', alpha=0.3)

        # 4. Success Rate by Discretization
        ax4 = fig.add_subplot(gs[1, 0])
        df['disc'] = df['num_steps_x'].astype(str) + 'x' + df['num_steps_y'].astype(str) + 'x' + df['num_steps_theta'].astype(str)
        success_by_disc = df.groupby(['disc', 'allowReopening'])['success'].mean().unstack()
        success_by_disc.plot(kind='bar', ax=ax4, color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black')
        ax4.set_title('Success Rate by Discretization', fontweight='bold', fontsize=11)
        ax4.set_ylabel('Success Rate')
        ax4.set_xlabel('Grid Resolution')
        ax4.legend(['Without Reopening', 'With Reopening'], loc='lower right')
        ax4.set_ylim([0, 1.1])
        ax4.grid(axis='y', alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 5. Path Cost Improvement
        ax5 = fig.add_subplot(gs[1, 1])
        if len(comparison_df) > 0:
            cost_imp = comparison_df.groupby('w')['path_cost_improvement_pct'].mean()
            colors = ['#2ecc71' if x < 0 else '#e74c3c' for x in cost_imp.values]
            ax5.bar(cost_imp.index, cost_imp.values, color=colors, alpha=0.7, edgecolor='black')
            ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax5.set_title('Path Cost Improvement by Weight', fontweight='bold', fontsize=11)
            ax5.set_ylabel('Improvement (%)\n(negative = better)')
            ax5.set_xlabel('Weight (w)')
            ax5.grid(axis='y', alpha=0.3)

        # 6. Time Overhead
        ax6 = fig.add_subplot(gs[1, 2])
        if len(comparison_df) > 0:
            time_overhead = comparison_df.groupby('w')['time_overhead_pct'].mean()
            colors = ['#2ecc71' if x < 0 else '#e74c3c' for x in time_overhead.values]
            ax6.bar(time_overhead.index, time_overhead.values, color=colors, alpha=0.7, edgecolor='black')
            ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax6.set_title('Time Overhead by Weight', fontweight='bold', fontsize=11)
            ax6.set_ylabel('Overhead (%)\n(positive = slower)')
            ax6.set_xlabel('Weight (w)')
            ax6.grid(axis='y', alpha=0.3)

        # 7. Nodes Explored by Configuration
        ax7 = fig.add_subplot(gs[2, 0])
        nodes_by_disc = df.groupby(['disc', 'allowReopening'])['num_nodes_in_graph'].mean().unstack()
        nodes_by_disc.plot(kind='bar', ax=ax7, color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black')
        ax7.set_title('Avg Nodes Explored by Discretization', fontweight='bold', fontsize=11)
        ax7.set_ylabel('Number of Nodes')
        ax7.set_xlabel('Grid Resolution')
        ax7.legend(['Without Reopening', 'With Reopening'])
        ax7.grid(axis='y', alpha=0.3)
        plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 8. Path Length Improvement Distribution
        ax8 = fig.add_subplot(gs[2, 1])
        if len(comparison_df) > 0 and 'path_length_improvement_pct' in comparison_df.columns:
            valid_improvements = comparison_df[comparison_df['success_without'] & comparison_df['success_with']]['path_length_improvement_pct']
            if len(valid_improvements) > 0:
                ax8.hist(valid_improvements, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
                ax8.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No change')
                ax8.set_title('Path Length Improvement Distribution', fontweight='bold', fontsize=11)
                ax8.set_xlabel('Improvement (%)\n(negative = shorter path)')
                ax8.set_ylabel('Frequency')
                ax8.legend()
                ax8.grid(axis='y', alpha=0.3)

        # 9. Efficiency: Nodes per Second
        ax9 = fig.add_subplot(gs[2, 2])
        efficiency = df.groupby(['w', 'allowReopening'])['nodes_per_second'].mean().unstack()
        efficiency.plot(kind='bar', ax=ax9, color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black')
        ax9.set_title('Planning Efficiency by Weight', fontweight='bold', fontsize=11)
        ax9.set_ylabel('Nodes per Second')
        ax9.set_xlabel('Weight (w)')
        ax9.legend(['Without Reopening', 'With Reopening'])
        ax9.grid(axis='y', alpha=0.3)

        # 10. Benchmark-specific comparison
        ax10 = fig.add_subplot(gs[3, 0])
        success_by_bench = df.groupby(['benchmark_name', 'allowReopening'])['success'].mean().unstack()
        success_by_bench.plot(kind='bar', ax=ax10, color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black')
        ax10.set_title('Success Rate by Benchmark', fontweight='bold', fontsize=11)
        ax10.set_ylabel('Success Rate')
        ax10.set_xlabel('Benchmark')
        ax10.legend(['Without Reopening', 'With Reopening'])
        ax10.set_ylim([0, 1.1])
        ax10.grid(axis='y', alpha=0.3)
        plt.setp(ax10.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 11. Time vs Quality Trade-off
        ax11 = fig.add_subplot(gs[3, 1])
        if len(success_df) > 0:
            for reopen in [False, True]:
                subset = success_df[success_df['allowReopening'] == reopen]
                if len(subset) > 0:
                    label = 'With Reopening' if reopen else 'Without Reopening'
                    color = '#2ecc71' if reopen else '#e74c3c'
                    ax11.scatter(subset['planning_time'], subset['path_cost'],
                               alpha=0.6, s=50, label=label, color=color, edgecolor='black')
            ax11.set_title('Time vs Path Cost Trade-off', fontweight='bold', fontsize=11)
            ax11.set_xlabel('Planning Time (s)')
            ax11.set_ylabel('Path Cost')
            ax11.legend()
            ax11.grid(alpha=0.3)

        # 12. Summary Statistics Table
        ax12 = fig.add_subplot(gs[3, 2])
        ax12.axis('tight')
        ax12.axis('off')

        summary_stats = []
        for reopen in [False, True]:
            subset = df[df['allowReopening'] == reopen]
            success_subset = subset[subset['success'] == True]
            label = 'With Reopening' if reopen else 'Without Reopening'

            stats = [
                label,
                f"{subset['success'].mean():.1%}",
                f"{success_subset['path_length'].mean():.1f}" if len(success_subset) > 0 else "N/A",
                f"{success_subset['path_cost'].mean():.2f}" if len(success_subset) > 0 else "N/A",
                f"{subset['planning_time'].mean():.3f}s",
                f"{subset['num_nodes_in_graph'].mean():.0f}"
            ]
            summary_stats.append(stats)

        table = ax12.table(cellText=summary_stats,
                          colLabels=['Configuration', 'Success\nRate', 'Avg Path\nLength',
                                    'Avg Path\nCost', 'Avg Time', 'Avg Nodes'],
                          cellLoc='center',
                          loc='center',
                          bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header
        for i in range(6):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Style rows
        colors = ['#e74c3c', '#2ecc71']
        for i in range(1, 3):
            for j in range(6):
                table[(i, j)].set_facecolor(colors[i-1])
                table[(i, j)].set_alpha(0.3)

        ax12.set_title('Summary Statistics', fontweight='bold', fontsize=11, pad=20)

        plt.suptitle('Comprehensive A* Reopening Analysis - 3DoF Shape Robot',
                     fontsize=16, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Visualization saved to: {save_path}")

        plt.show()

        return fig


def main():
    """Run comprehensive analysis"""
    print("Starting Comprehensive Reopening Analysis...")
    print("This will test multiple combinations of:")
    print("  - Heuristic weights (w)")
    print("  - Grid discretizations (num_steps)")
    print("  - With and without reopening")
    print()

    analyzer = ReopeningAnalyzer()

    # Define test parameters
    w_values = [0.5, 1.0, 1.5, 2.0]
    num_steps_configs = [
        [15, 15, 24],  # Coarse
        [20, 20, 36],  # Medium
        [25, 25, 36],  # Fine
        [30, 30, 48],  # Very fine
    ]

    # Run comprehensive analysis
    results_df = analyzer.run_comprehensive_analysis(
        benchmarks=ts.benchList,
        w_values=w_values,
        num_steps_configs=num_steps_configs,
        verbose=True
    )

    # Save detailed results
    results_df.to_csv("comprehensive_reopening_results.csv", index=False)
    print(f"\n[OK] Detailed results saved to: comprehensive_reopening_results.csv")

    # Compute comparative metrics
    print("\nComputing comparative metrics...")
    comparison_df = analyzer.compute_comparative_metrics(results_df)
    comparison_df.to_csv("reopening_comparison_metrics.csv", index=False)
    print(f"[OK] Comparison metrics saved to: reopening_comparison_metrics.csv")

    # Display key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    print("\n1. OVERALL SUCCESS RATES:")
    overall_success = results_df.groupby('allowReopening')['success'].mean()
    for reopen, rate in overall_success.items():
        label = "WITH reopening" if reopen else "WITHOUT reopening"
        print(f"   {label}: {rate:.1%}")

    print("\n2. AVERAGE PATH IMPROVEMENTS (successful cases only):")
    successful_comparisons = comparison_df[comparison_df['success_with'] & comparison_df['success_without']]
    if len(successful_comparisons) > 0:
        avg_path_improvement = successful_comparisons['path_length_improvement_pct'].mean()
        avg_cost_improvement = successful_comparisons['path_cost_improvement_pct'].mean()
        print(f"   Path length improvement: {avg_path_improvement:.2f}%")
        print(f"   Path cost improvement: {avg_cost_improvement:.2f}%")

    print("\n3. COMPUTATIONAL OVERHEAD:")
    avg_time_overhead = comparison_df['time_overhead_pct'].mean()
    print(f"   Average time overhead: {avg_time_overhead:.2f}%")

    print("\n4. BEST CONFIGURATIONS FOR REOPENING:")
    if len(successful_comparisons) > 0:
        # Find configs where reopening helps most
        best_configs = successful_comparisons.nsmallest(5, 'path_cost_improvement_pct')
        print("   Top 5 configurations (by path cost improvement):")
        for idx, row in best_configs.iterrows():
            print(f"   - w={row['w']}, grid={row['num_steps']}: "
                  f"{row['path_cost_improvement_pct']:.2f}% cost improvement, "
                  f"{row['time_overhead_pct']:.1f}% time overhead")

    print("\n5. WHEN TO USE REOPENING:")
    print("   Reopening is beneficial when:")

    # Analyze by weight
    by_weight = analyzer.analyze_by_weight(results_df)
    print("\n   By Weight Analysis:")
    for w in w_values:
        try:
            success_without = by_weight.loc[(w, False), 'success']
            success_with = by_weight.loc[(w, True), 'success']
            time_without = by_weight.loc[(w, False), 'planning_time']
            time_with = by_weight.loc[(w, True), 'planning_time']

            if success_with > success_without:
                print(f"   - w={w}: Reopening improves success rate "
                      f"({success_without:.1%} -> {success_with:.1%})")
        except KeyError:
            continue

    # Create visualizations
    print("\n" + "="*80)
    print("Creating comprehensive visualizations...")
    print("="*80)

    analyzer.create_comprehensive_visualizations(
        results_df,
        comparison_df,
        save_path="comprehensive_reopening_analysis.png"
    )

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. comprehensive_reopening_results.csv - All test results")
    print("  2. reopening_comparison_metrics.csv - Comparative metrics")
    print("  3. comprehensive_reopening_analysis.png - Visualization suite")

    return results_df, comparison_df


if __name__ == "__main__":
    results_df, comparison_df = main()
