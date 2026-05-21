from utils import generate_ecg_data, create_animation, save_snr_cluster_plot

if __name__ == "__main__":
    df = generate_ecg_data()
    create_animation(df)
    save_snr_cluster_plot(df)
