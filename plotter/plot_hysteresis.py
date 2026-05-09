import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_hysteresis(csv_path, title, output_path):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # branch 0 is descending (+h -> -h)
    # branch 1 is ascending (-h -> +h)
    branch0 = df[df['branch'] == 0]
    branch1 = df[df['branch'] == 1]

    plt.figure(figsize=(8, 6))
    
    # Plot descending branch
    plt.plot(branch0['h'], branch0['M'], label='Descending (+h to -h)', color='red', marker='o', markersize=3, linestyle='-')
    
    # Plot ascending branch
    plt.plot(branch1['h'], branch1['M'], label='Ascending (-h to +h)', color='blue', marker='x', markersize=3, linestyle='--')

    plt.title(title)
    plt.xlabel('Magnetic Field (h)')
    plt.ylabel('Magnetization (M)')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot hysteresis from CSV')
    parser.add_argument('--ising', default='../ising_hysteresis.csv', help='Path to Ising CSV')
    parser.add_argument('--ising_gpu', default='../ising_gpu_hysteresis.csv', help='Path to GPU Ising CSV')
    parser.add_argument('--xy', default='../xy_hysteresis.csv', help='Path to XY CSV')
    
    args = parser.parse_args()
    
    plot_hysteresis(args.ising, 'Ising Model Hysteresis', 'ising_plot.png')
    plot_hysteresis(args.ising_gpu, 'GPU Ising Model Hysteresis', 'ising_gpu_plot.png')
    plot_hysteresis(args.xy, 'XY Model Hysteresis', 'xy_plot.png')
