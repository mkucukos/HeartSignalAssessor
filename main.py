from utils import generate_ecg_data, create_animation

if __name__ == "__main__":
    df = generate_ecg_data()
    create_animation(df)
