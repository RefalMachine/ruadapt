import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

def smooth(data, window_size):
    if not data or window_size < 2:
        return data
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        smoothed.append(sum(data[start:i+1]) / (i - start + 1))
    return smoothed

def main(args):
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.tab10.colors
    
    for idx, filepath in enumerate(args.files):
        path = Path(filepath)
        if not path.exists():
            print(f"Warning: File not found: {path}")
            continue
            
        with open(path, 'r') as f:
            data = json.load(f)
        
        val_steps = data.get('val_steps', [])
        
        # Dynamically fetch metric from metrics dictionary
        metric_key_map = {
            'mse': 'mse_dist',
            'cos_dist': 'cosine_dist',
            'rel_dist': 'rel_dist',
            'mrr': 'mrr'
        }
        internal_key = metric_key_map.get(args.plot_metric, args.plot_metric)
        
        val_losses = []
        if 'metrics' in data and internal_key in data['metrics']:
            val_losses = data['metrics'][internal_key]
        else:
            # Fallback for old logs
            if args.plot_metric == 'mse':
                val_losses = data.get('val_losses_step', [])
            elif args.plot_metric == 'cos_dist':
                val_losses = data.get('val_cos_dist_step', [])
                
        # Train losses are available for the primary composite loss
        if args.plot_metric == 'loss':
            val_losses = data.get('val_losses_step', [])
            train_losses = data.get('train_losses_step', [])
        else:
            train_losses = []
        
        # Cleanup label for legend
        label = path.stem.replace('_logs', '').replace('h1_2_bpe_training_logs_', 'old_v0_')
        color = colors[idx % len(colors)]
        
        if val_steps and val_losses:
            # Filter validation points
            v_filtered = [(s, l) for s, l in zip(val_steps, val_losses) if s >= args.skip_steps]
            if v_filtered:
                f_steps, f_losses = zip(*v_filtered)
                plt.plot(f_steps, f_losses, label=f"{label} (val {args.plot_metric})", marker='o', markersize=4, color=color, linewidth=2)
            
        if args.show_train and train_losses:
            max_val_step = max(val_steps) if val_steps else len(train_losses)
            interval = max(1, max_val_step // len(train_losses))
            train_steps = [i * interval for i in range(len(train_losses))]
            
            # Filter train points
            t_filtered = [(s, l) for s, l in zip(train_steps, train_losses) if s >= args.skip_steps]
            if t_filtered:
                f_t_steps, f_t_losses = zip(*t_filtered)
                
                window = max(1, len(f_t_losses) // 50)
                smoothed_train = smooth(f_t_losses, window)
                
                plt.plot(f_t_steps, smoothed_train, label=f"{label} (train_smooth)", color=color, alpha=0.3, linestyle='--')

    title_map = {
        'mse': 'Training and Validation MSE Loss',
        'cos_dist': 'Training and Validation Cosine Distance',
        'rel_dist': 'Validation Relative Distance (RelDist)',
        'mrr': 'Validation Mean Reciprocal Rank (MRR)',
        'loss': 'Composite Loss (Objective)'
    }
    plt.title(title_map.get(args.plot_metric, f'Validation {args.plot_metric.upper()}'))
    plt.xlabel('Global Step')
    
    ylabel_map = {
        'mse': 'MSE Loss',
        'cos_dist': 'Cosine Distance',
        'rel_dist': 'Relative Distance',
        'mrr': 'MRR',
        'loss': 'Composite Loss'
    }
    plt.ylabel(ylabel_map.get(args.plot_metric, 'Metric'))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    if args.log_scale:
        plt.yscale('log')
    #elif args.plot_metric == 'mse':
    #    plt.ylim(bottom=0.0001)
    #elif args.plot_metric == 'loss':
    #    plt.ylim(bottom=0.0)
        
    out_path = Path('results') / args.output
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs='+', help="Paths to JSON log files to plot")
    parser.add_argument("--output", default="loss_comparison.png", help="Output filename in results dir")
    parser.add_argument("--show-train", action="store_true", help="Plot smoothed training loss as well")
    parser.add_argument("--log-scale", action="store_true", help="Use logarithmic scale for Y axis")
    parser.add_argument("--skip-steps", type=int, default=0, help="Skip the first N steps to zoom in on the tail")
    parser.add_argument("--plot-metric", choices=['loss', 'mse', 'cos_dist', 'rel_dist', 'mrr'], default='mse', help="Which metric to plot")
    args = parser.parse_args()
    main(args)