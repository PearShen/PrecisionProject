"""
Operator trace formatting and output utilities.
Provides various formats for displaying and analyzing captured operator information.
"""

import json
import csv
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from operator_capture import EnhancedOperatorInfo


class OperatorTraceFormatter:
    """
    Formats and exports operator traces in various formats.

    Supports:
    - JSON reports
    - CSV exports
    - HTML reports with visualizations
    - Plain text summaries
    - Performance analysis charts
    """

    def __init__(self, traces: List[EnhancedOperatorInfo]):
        """
        Initialize with operator traces.

        Args:
            traces: List of enhanced operator traces
        """
        self.traces = traces
        self.traces_dicts = self._convert_traces_to_dicts()

    def _convert_traces_to_dicts(self) -> List[Dict[str, Any]]:
        """Convert EnhancedOperatorInfo objects to dictionaries"""
        return [
            {
                'iteration': trace.iteration,
                'model_name': trace.model_name,
                'operator_name': trace.operator_name,
                'operator_type': trace.operator_type,
                'module_path': trace.module_path,
                'layer_name': trace.layer_name,
                'call_site': trace.call_site,
                'input_shapes': trace.input_shapes,
                'output_shapes': trace.output_shapes,
                'input_dtypes': trace.input_dtypes,
                'output_dtypes': trace.output_dtypes,
                'execution_time_ms': trace.execution_time_ms,
                'memory_alloc_mb': trace.memory_alloc_mb,
                'arguments': trace.arguments,
                'num_tensor_inputs': len(trace.tensor_inputs),
                'num_tensor_outputs': len(trace.tensor_outputs),
                'tensor_input_shapes': [list(t.shape) for t in trace.tensor_inputs],
                'tensor_input_dtypes': [str(t.dtype) for t in trace.tensor_inputs],
                'tensor_output_shapes': [list(t.shape) for t in trace.tensor_outputs],
                'tensor_output_dtypes': [str(t.dtype) for t in trace.tensor_outputs],
                'timestamp': trace.timestamp,
                'thread_id': trace.thread_id,
                'context_info': trace.context_info
            }
            for trace in self.traces
        ]

    def generate_summary_report(self, output_path: str) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report.

        Args:
            output_path: Path to save the summary report

        Returns:
            Dictionary containing the summary statistics
        """
        if not self.traces:
            return {'error': 'No traces available'}

        # Basic statistics
        summary = {
            'model_info': {
                'model_name': self.traces[0].model_name,
                'total_traces': len(self.traces),
                'unique_operators': len(set(t.operator_name for t in self.traces)),
                'operator_types': list(set(t.operator_type for t in self.traces)),
                'iterations': max(t.iteration for t in self.traces) + 1,
                'time_range': {
                    'start': min(t.timestamp for t in self.traces),
                    'end': max(t.timestamp for t in self.traces),
                    'duration_ms': (max(t.timestamp for t in self.traces) - min(t.timestamp for t in self.traces)) * 1000
                }
            },
            'operator_statistics': self._get_operator_statistics(),
            'performance_analysis': self._get_performance_analysis(),
            'memory_analysis': self._get_memory_analysis(),
            'execution_flow': self._get_execution_flow_analysis(),
            'unique_layers': list(set(t.layer_name for t in self.traces if t.layer_name != 'unknown'))
        }

        # Save summary to JSON
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        return summary

    def _get_operator_statistics(self) -> Dict[str, Any]:
        """Get detailed operator statistics"""
        stats = {}

        # Count by operator name
        operator_counts = {}
        for trace in self.traces:
            name = trace.operator_name
            operator_counts[name] = operator_counts.get(name, 0) + 1
        stats['operator_counts'] = dict(sorted(operator_counts.items(), key=lambda x: x[1], reverse=True))

        # Count by operator type
        type_counts = {}
        for trace in self.traces:
            op_type = trace.operator_type
            type_counts[op_type] = type_counts.get(op_type, 0) + 1
        stats['type_counts'] = type_counts

        # Execution time by operator
        exec_time_by_op = {}
        for trace in self.traces:
            name = trace.operator_name
            exec_time_by_op[name] = exec_time_by_op.get(name, 0) + trace.execution_time_ms
        stats['total_execution_time_by_operator'] = dict(sorted(exec_time_by_op.items(), key=lambda x: x[1], reverse=True))

        # Memory allocation by operator
        memory_by_op = {}
        for trace in self.traces:
            name = trace.operator_name
            memory_by_op[name] = memory_by_op.get(name, 0) + trace.memory_alloc_mb
        stats['total_memory_by_operator'] = dict(sorted(memory_by_op.items(), key=lambda x: x[1], reverse=True))

        return stats

    def _get_performance_analysis(self) -> Dict[str, Any]:
        """Get performance analysis"""
        if not self.traces:
            return {}

        exec_times = [t.execution_time_ms for t in self.traces]
        times_by_op = {}
        for trace in self.traces:
            name = trace.operator_name
            if name not in times_by_op:
                times_by_op[name] = []
            times_by_op[name].append(trace.execution_time_ms)

        # Calculate statistics
        perf_analysis = {
            'total_execution_time_ms': sum(exec_times),
            'average_execution_time_ms': sum(exec_times) / len(exec_times),
            'max_execution_time_ms': max(exec_times),
            'min_execution_time_ms': min(exec_times),
            'median_execution_time_ms': sorted(exec_times)[len(exec_times)//2],
            'slowest_operations': [
                {
                    'operator': trace.operator_name,
                    'execution_time_ms': trace.execution_time_ms,
                    'layer': trace.layer_name,
                    'call_site': trace.call_site
                }
                for trace in sorted(self.traces, key=lambda x: x.execution_time_ms, reverse=True)[:10]
            ]
        }

        # Average time per operator
        perf_analysis['average_time_by_operator'] = {
            op: {
                'avg_ms': sum(times) / len(times),
                'min_ms': min(times),
                'max_ms': max(times),
                'count': len(times)
            }
            for op, times in times_by_op.items()
        }

        return perf_analysis

    def _get_memory_analysis(self) -> Dict[str, Any]:
        """Get memory usage analysis"""
        if not self.traces:
            return {}

        memory_allocs = [t.memory_alloc_mb for t in self.traces if t.memory_alloc_mb > 0]
        if not memory_allocs:
            return {'message': 'No memory allocation data available'}

        memory_by_op = {}
        for trace in self.traces:
            if trace.memory_alloc_mb > 0:
                name = trace.operator_name
                if name not in memory_by_op:
                    memory_by_op[name] = []
                memory_by_op[name].append(trace.memory_alloc_mb)

        memory_analysis = {
            'total_memory_allocated_mb': sum(memory_allocs),
            'average_memory_per_op_mb': sum(memory_allocs) / len(memory_allocs),
            'max_memory_per_op_mb': max(memory_allocs),
            'memory_heavy_operations': [
                {
                    'operator': trace.operator_name,
                    'memory_alloc_mb': trace.memory_alloc_mb,
                    'layer': trace.layer_name,
                    'call_site': trace.call_site
                }
                for trace in sorted(self.traces, key=lambda x: x.memory_alloc_mb, reverse=True)[:10]
            ]
        }

        if memory_by_op:
            memory_analysis['memory_by_operator'] = {
                op: {
                    'total_mb': sum(memorys),
                    'avg_mb': sum(memorys) / len(memorys),
                    'max_mb': max(memorys),
                    'count': len(memorys)
                }
                for op, memorys in memory_by_op.items()
            }

        return memory_analysis

    def _get_execution_flow_analysis(self) -> Dict[str, Any]:
        """Get execution flow analysis"""
        if not self.traces:
            return {}

        # Analyze by iteration
        iterations = {}
        for trace in self.traces:
            iter_num = trace.iteration
            if iter_num not in iterations:
                iterations[iter_num] = {
                    'operators': [],
                    'start_time': trace.timestamp,
                    'end_time': trace.timestamp,
                    'total_ops': 0
                }
            iterations[iter_num]['operators'].append(trace.operator_name)
            iterations[iter_num]['end_time'] = max(iterations[iter_num]['end_time'], trace.timestamp)
            iterations[iter_num]['total_ops'] += 1

        flow_analysis = {
            'iterations_count': len(iterations),
            'operators_per_iteration': {i: data['total_ops'] for i, data in iterations.items()},
            'total_duration_ms': sum(data['end_time'] - data['start_time'] for data in iterations.values()) * 1000
        }

        # Copy with operator type flow
        flow_by_type = {}
        for trace in self.traces:
            op_type = trace.operator_type
            if op_type not in flow_by_type:
                flow_by_type[op_type] = {
                    'count': 0,
                    'operators': set(),
                    'total_time_ms': 0,
                    'total_memory_mb': 0
                }
            flow_by_type[op_type]['count'] += 1
            flow_by_type[op_type]['operators'].add(trace.operator_name)
            flow_by_type[op_type]['total_time_ms'] += trace.execution_time_ms
            flow_by_type[op_type]['total_memory_mb'] += trace.memory_alloc_mb

        # Convert sets to lists for serialization
        for op_type in flow_by_type:
            flow_by_type[op_type]['operators'] = list(flow_by_type[op_type]['operators'])

        flow_analysis['flow_by_operation_type'] = flow_by_type

        return flow_analysis

    def export_to_csv(self, output_path: str) -> None:
        """Export traces to CSV file"""
        df_traces = []

        for trace_dict in self.traces_dicts:
            # Flatten nested structures for CSV
            csv_row = {
                'iteration': trace_dict['iteration'],
                'model_name': trace_dict['model_name'],
                'operator_name': trace_dict['operator_name'],
                'operator_type': trace_dict['operator_type'],
                'module_path': trace_dict['module_path'],
                'layer_name': trace_dict['layer_name'],
                'call_site': trace_dict['call_site'],
                'execution_time_ms': trace_dict['execution_time_ms'],
                'memory_alloc_mb': trace_dict['memory_alloc_mb'],
                'num_tensor_inputs': trace_dict['num_tensor_inputs'],
                'num_tensor_outputs': trace_dict['num_tensor_outputs'],
                'timestamp': trace_dict['timestamp'],
                'thread_id': trace_dict['thread_id']
            }

            # Add input/output shapes as separate columns
            for i, shape in enumerate(trace_dict['input_shapes']):
                csv_row[f'input_shape_{i}'] = str(shape)

            for i, shape in enumerate(trace_dict['output_shapes']):
                csv_row[f'output_shape_{i}'] = str(shape)

            # Add input/output dtypes
            for i, dtype in enumerate(trace_dict['input_dtypes']):
                csv_row[f'input_dtype_{i}'] = dtype

            for i, dtype in enumerate(trace_dict['output_dtypes']):
                csv_row[f'output_dtype_{i}'] = dtype

            df_traces.append(csv_row)

        df = pd.DataFrame(df_traces)
        df.to_csv(output_path, index=False)

    def generate_html_report(self, output_path: str, include_charts: bool = True) -> None:
        """
        Generate a comprehensive HTML report with visualizations.

        Args:
            output_path: Path to save the HTML report
            include_charts: Whether to include interactive charts
        """
        # Generate summary data
        summary = self.generate_summary_report(output_path.replace('.html', '_summary.json'))

        # Generate charts if requested
        chart_data = {}
        if include_charts:
            chart_data = self._generate_charts(output_path.replace('.html', '_charts'))

        # HTML template
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Operator Trace Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; color: #333; border-bottom: 2px solid #007bff; padding-bottom: 20px; }}
                .section {{ margin: 30px 0; }}
                .section h2 {{ color: #007bff; border-left: 4px solid #007bff; padding-left: 10px; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
                .stat-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border: 1px solid #e9ecef; }}
                .stat-card h3 {{ margin: 0 0 10px 0; color: #495057; }}
                .stat-card .value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #007bff; color: white; }}
                .chart {{ margin: 20px 0; text-align: center; }}
                .footnote {{ font-size: 12px; color: #6c757d; margin-top: 30px; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔍 Operator Trace Analysis Report</h1>
                    <p>Comprehensive analysis of {summary['model_info']['total_traces']} operator executions</p>
                </div>

                <div class="section">
                    <h2>📊 Model Overview</h2>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <h3>Model Name</h3>
                            <div class="value">{summary['model_info']['model_name']}</div>
                        </div>
                        <div class="stat-card">
                            <h3>Total Operators</h3>
                            <div class="value">{summary['model_info']['total_traces']}</div>
                        </div>
                        <div class="stat-card">
                            <h3>Unique Operators</h3>
                            <div class="value">{summary['model_info']['unique_operators']}</div>
                        </div>
                        <div class="stat-card">
                            <h3>Iterations</h3>
                            <div class="value">{summary['model_info']['iterations']}</div>
                        </div>
                    </div>
                </div>

                <div class="section">
                    <h2>⚡ Performance Analysis</h2>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <h3>Total Execution Time</h3>
                            <div class="value">{summary['performance_analysis']['total_execution_time_ms']:.2f} ms</div>
                        </div>
                        <div class="stat-card">
                            <h3>Average per Operator</h3>
                            <div class="value">{summary['performance_analysis']['average_execution_time_ms']:.2f} ms</div>
                        </div>
                        <div class="stat-card">
                            <h3>Slowest Operation</h3>
                            <div class="value">{summary['performance_analysis']['max_execution_time_ms']:.2f} ms</div>
                        </div>
                        <div class="stat-card">
                            <h3>Total Memory Allocated</h3>
                            <div class="value">{summary.get('memory_analysis', {}).get('total_memory_allocated_mb', 0):.2f} MB</div>
                        </div>
                    </div>
                </div>

                <div class="section">
                    <h2>🔧 Top Operators by Frequency</h2>
                    <table>
                        <tr><th>Operator</th><th>Count</th><th>Total Time (ms)</th><th>Total Memory (MB)</th></tr>
                        {self._format_top_operators_table(summary)}
                    </table>
                </div>

                <div class="section">
                    <h2>🐌 Slowest Operations</h2>
                    <table>
                        <tr><th>Operator</th><th>Layer</th><th>Execution Time (ms)</th><th>Memory (MB)</th></tr>
                        {self._format_slowest_operations_table(summary)}
                    </table>
                </div>

                {self._format_charts_section(chart_data) if include_charts else ''}

                <div class="footnote">
                    <p>Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | Powered by PrecisionProject 🔬</p>
                </div>
            </div>
        </body>
        </html>
        """

        with open(output_path, 'w') as f:
            f.write(html_content)

    def _format_top_operators_table(self, summary: Dict[str, Any]) -> str:
        """Format top operators table rows"""
        rows = []
        operator_stats = summary['operator_statistics']

        # Combine data from different sources
        for op_name in list(operator_stats['operator_counts'].keys())[:10]:
            count = operator_stats['operator_counts'][op_name]
            exec_time = operator_stats['total_execution_time_by_operator'].get(op_name, 0)
            memory = operator_stats['total_memory_by_operator'].get(op_name, 0)

            rows.append(f"""
                <tr>
                    <td><strong>{op_name}</strong></td>
                    <td>{count}</td>
                    <td>{exec_time:.2f}</td>
                    <td>{memory:.2f}</td>
                </tr>
            """)

        return ''.join(rows)

    def _format_slowest_operations_table(self, summary: Dict[str, Any]) -> str:
        """Format slowest operations table rows"""
        rows = []
        for slow_op in summary['performance_analysis']['slowest_operations'][:10]:
            rows.append(f"""
                <tr>
                    <td><strong>{slow_op['operator']}</strong></td>
                    <td>{slow_op['layer']}</td>
                    <td>{slow_op['execution_time_ms']:.2f}</td>
                    <td>0.00</td>
                </tr>
            """)

        return ''.join(rows)

    def _format_charts_section(self, chart_data: Dict[str, str]) -> str:
        """Format charts section with embedded images"""
        if not chart_data:
            return ""

        charts_html = """
        <div class="section">
            <h2>📈 Visual Analytics</h2>
        """

        for chart_name, chart_path in chart_data.items():
            charts_html += f"""
            <div class="chart">
                <h3>{chart_name}</h3>
                <img src="{Path(chart_path).name}" alt="{chart_name}" style="max-width: 100%; height: auto;">
            </div>
            """

        charts_html += "</div>"
        return charts_html

    def _generate_charts(self, output_dir: str) -> Dict[str, str]:
        """Generate charts and return paths to saved images"""
        Path(output_dir).mkdir(exist_ok=True)
        chart_paths = {}

        if not self.traces:
            return chart_paths

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # 1. Operator frequency bar chart
        operator_names = [t.operator_name for t in self.traces]
        operator_counts = pd.Series(operator_names).value_counts().head(15)

        plt.figure(figsize=(12, 6))
        operator_counts.plot(kind='bar')
        plt.title('Top 15 Operators by Frequency')
        plt.xlabel('Operator Name')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()

        freq_chart_path = Path(output_dir) / 'operator_frequency.png'
        plt.savefig(freq_chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths['Operator Frequency'] = str(freq_chart_path)

        # 2. Execution time distribution
        exec_times = [t.execution_time_ms for t in self.traces]

        plt.figure(figsize=(10, 6))
        plt.hist(exec_times, bins=50, alpha=0.7, edgecolor='black')
        plt.title('Execution Time Distribution')
        plt.xlabel('Execution Time (ms)')
        plt.ylabel('Frequency')
        plt.tight_layout()

        time_dist_path = Path(output_dir) / 'execution_time_distribution.png'
        plt.savefig(time_dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths['Execution Time Distribution'] = str(time_dist_path)

        # 3. Operator type pie chart
        operator_types = [t.operator_type for t in self.traces]
        type_counts = pd.Series(operator_types).value_counts()

        plt.figure(figsize=(8, 8))
        plt.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
        plt.title('Operators by Type')
        plt.tight_layout()

        type_pie_path = Path(output_dir) / 'operator_types.png'
        plt.savefig(type_pie_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths['Operators by Type'] = str(type_pie_path)

        # 4. Performance timeline (if multiple iterations)
        iterations = [t.iteration for t in self.traces]
        if len(set(iterations)) > 1:
            iteration_counts = pd.Series(iterations).value_counts().sort_index()

            plt.figure(figsize=(10, 6))
            iteration_counts.plot(kind='line', marker='o')
            plt.title('Operations per Iteration')
            plt.xlabel('Iteration Number')
            plt.ylabel('Number of Operations')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            timeline_path = Path(output_dir) / 'execution_timeline.png'
            plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
            plt.close()
            chart_paths['Execution Timeline'] = str(timeline_path)

        return chart_paths

    def generate_plain_text_summary(self) -> str:
        """Generate a plain text summary of the traces"""
        if not self.traces:
            return "No operator traces available."

        summary = self.generate_summary_report('/dev/null')

        text_output = f"""
🔍 OPERATOR TRACE ANALYSIS SUMMARY
{'='*50}

MODEL INFORMATION:
- Model Name: {summary['model_info']['model_name']}
- Total Traces: {summary['model_info']['total_traces']}
- Unique Operators: {summary['model_info']['unique_operators']}
- Operator Types: {', '.join(summary['model_info']['operator_types'])}
- Iterations: {summary['model_info']['iterations']}
- Duration: {summary['model_info']['time_range']['duration_ms']:.2f} ms

PERFORMANCE ANALYSIS:
- Total Execution Time: {summary['performance_analysis']['total_execution_time_ms']:.2f} ms
- Average per Operation: {summary['performance_analysis']['average_execution_time_ms']:.2f} ms
- Slowest Operation: {summary['performance_analysis']['max_execution_time_ms']:.2f} ms
- Fastest Operation: {summary['performance_analysis']['min_execution_time_ms']:.2f} ms

TOP 10 OPERATORS BY FREQUENCY:
{self._format_text_operator_frequency(summary)}

TOP 5 SLOWEST OPERATIONS:
{self._format_text_slowest_operations(summary)}

MEMORY ANALYSIS:
{self._format_text_memory_analysis(summary)}

EXECUTION FLOW:
{self._format_text_execution_flow(summary)}

Generate full report with: generate_html_report()
"""

        return text_output.strip()

    def _format_text_operator_frequency(self, summary: Dict[str, Any]) -> str:
        """Format operator frequency for text output"""
        lines = []
        for op_name, count in list(summary['operator_statistics']['operator_counts'].items())[:10]:
            exec_time = summary['operator_statistics']['total_execution_time_by_operator'].get(op_name, 0)
            lines.append(f"  - {op_name}: {count} calls, {exec_time:.2f} ms total")
        return '\n'.join(lines)

    def _format_text_slowest_operations(self, summary: Dict[str, Any]) -> str:
        """Format slowest operations for text output"""
        lines = []
        for i, slow_op in enumerate(summary['performance_analysis']['slowest_operations'][:5], 1):
            lines.append(f"  {i}. {slow_op['operator']} ({slow_op['layer']}): {slow_op['execution_time_ms']:.2f} ms")
        return '\n'.join(lines)

    def _format_text_memory_analysis(self, summary: Dict[str, Any]) -> str:
        """Format memory analysis for text output"""
        if 'memory_analysis' not in summary or 'total_memory_allocated_mb' not in summary['memory_analysis']:
            return "  No memory allocation data available."

        mem = summary['memory_analysis']
        return f"""
  - Total Memory Allocated: {mem['total_memory_allocated_mb']:.2f} MB
  - Average per Operation: {mem['average_memory_per_op_mb']:.2f} MB
  - Maximum per Operation: {mem['max_memory_per_op_mb']:.2f} MB
        """.strip()

    def _format_text_execution_flow(self, summary: Dict[str, Any]) -> str:
        """Format execution flow for text output"""
        flow = summary['execution_flow']
        lines = [
            f"  - Iterations: {flow['iterations_count']}",
            f"  - Total Duration: {flow['total_duration_ms']:.2f} ms",
            "  - Operations by Iteration:"
        ]
        for iter_num, count in flow['operators_per_iteration'].items():
            lines.append(f"    Iteration {iter_num}: {count} operators")
        return '\n'.join(lines)