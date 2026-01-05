import os
import sys
import subprocess

def run_static_analysis():
    print("Running Static Analysis...")
    
    # Distribution Plots
    print("Generating Token Density and Depth Distribution plots...")
    subprocess.run([sys.executable, "examples/visualization/static_analysis/distribution_plots.py"], check=True)
    
    # ViT Comparison
    print("Generating ViT Comparison plot...")
    subprocess.run([sys.executable, "examples/visualization/static_analysis/vit_comparison.py"], check=True)
    
    print("Static analysis complete. Check the output images in 'workspace/visualizations/static'.")

def print_manim_instructions():
    print("\n" + "="*50)
    print("MANIM ANIMATIONS")
    print("="*50)
    print("To generate the animations, you need to have Manim installed.")
    print("Install it via: pip install manim")
    print("\nRun the following commands (outputs will be in 'workspace/visualizations/media'):")
    print("1. Tokenization Animation:")
    print("   manim -pql --media_dir workspace/visualizations/media examples/visualization/manim_animations/tokenization.py AdaptiveTokenization")
    print("\n2. Attention Animation:")
    print("   manim -pql --media_dir workspace/visualizations/media examples/visualization/manim_animations/attention.py LCAAttention")
    print("="*50)

if __name__ == "__main__":
    # Ensure we are in the project root
    if not os.path.exists("examples/visualization"):
        print("Error: Please run this script from the project root directory.")
        sys.exit(1)
        
    run_static_analysis()
    print_manim_instructions()
