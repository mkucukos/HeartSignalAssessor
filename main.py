import os
from utils import generate_ecg_data, create_animation, save_snr_cluster_plot

os.makedirs("assets", exist_ok=True)

if __name__ == "__main__":
    df = generate_ecg_data()
    create_animation(df, output_stem="assets/ecg_analysis_animation")
    save_snr_cluster_plot(df, output_path="assets/snr_cluster_plot.png")
