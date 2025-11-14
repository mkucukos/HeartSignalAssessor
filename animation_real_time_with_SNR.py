import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import neurokit2 as nk
from scipy import stats

# Load model
model_path = "./model/"
model = tf.saved_model.load(model_path)
scaler = StandardScaler()

def get_ecg_features(ecg, time_in_sec, fs):
    """Extract ECG features including HR statistics and SNR"""
    try:
        b, a = butter(4, (0.25, 25), 'bandpass', fs=fs)
        ecg_filt = filtfilt(b, a, ecg, axis=0)
        ecg_cleaned = nk.ecg_clean(ecg_filt, sampling_rate=fs)
        instant_peaks, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs, method="engzeemod2012")
    except Exception as e:
        raise ValueError("Error processing ECG signal: " + str(e))

    rr_times = time_in_sec[rpeaks['ECG_R_Peaks']]
    if len(rr_times) == 0:
        raise ValueError("No R-peaks detected in ECG signal.")
    
    # Calculate heart rate statistics
    d_rr = np.diff(rr_times)
    heart_rate = 60 / d_rr
    if heart_rate.size == 0:
        raise ValueError("Error computing heart rate from ECG signal.")
    
    valid_heart_rate = heart_rate[~np.isnan(heart_rate)]
    z_scores = np.abs(stats.zscore(valid_heart_rate))
    z_score_threshold = 2.0
    heart_rate = valid_heart_rate[z_scores <= z_score_threshold]

    hr_mean = np.nanmean(heart_rate)
    hr_min = np.nanmin(heart_rate)
    hr_max = np.nanmax(heart_rate)
    
    # Calculate HRV
    d_rr_ms = 1000 * d_rr
    d_d_rr_ms = np.diff(d_rr_ms)
    valid_d_d_rr_ms = d_d_rr_ms[~np.isnan(d_d_rr_ms)] 
    z_scores = np.abs(stats.zscore(valid_d_d_rr_ms))
    d_d_rr_ms = valid_d_d_rr_ms[z_scores <= z_score_threshold]
    heart_rate_variability = np.sqrt(np.nanmean(np.square(d_d_rr_ms)))

    # Calculate SNR
    ecg_with_rr_intervals = []
    ecg_with_rr_intervals_cleaned = []

    for rr_interval in rr_times:
        start_time = rr_interval - 0.05
        end_time = rr_interval + 0.05
        indices = np.where((time_in_sec >= start_time) & (time_in_sec <= end_time))[0]
        indices = indices[(indices >= 0) & (indices < len(ecg))]

        if len(indices) > 0:
            ecg_with_rr_intervals.extend(ecg[indices])
            ecg_with_rr_intervals_cleaned.extend(ecg_cleaned[indices])

    ecg_with_rr_intervals = np.array(ecg_with_rr_intervals)
    ecg_with_rr_intervals_cleaned = np.array(ecg_with_rr_intervals_cleaned)

    signal_power = np.var(ecg_with_rr_intervals)
    noise_power = np.var(ecg_with_rr_intervals - ecg_with_rr_intervals_cleaned)
    snr_values = 10 * np.log10(signal_power / noise_power)

    return np.array([hr_mean, hr_max, hr_min, heart_rate_variability, snr_values])

def get_noise_std(frame_count):
    """Progressive noise schedule for simulation"""
    if frame_count < 20:
        return 0.01
    elif frame_count < 50:
        return 0.05
    elif frame_count < 75:
        return 0.15
    elif frame_count < 100:
        return 0.30
    elif frame_count < 125:
        return 0.45
    elif frame_count < 150:
        return 0.60
    elif frame_count < 175:
        return 0.80
    elif frame_count < 200:
        return 1.00
    elif frame_count < 450:
        return 1.25
    elif frame_count < 500:
        return 1.50
    elif frame_count < 550:
        return 1.75
    elif frame_count < 600:
        return 2.00
    elif frame_count < 650:
        return 1.80
    elif frame_count < 700:
        return 1.60
    elif frame_count < 750:
        return 1.40
    elif frame_count < 800:
        return 1.20
    elif frame_count < 850:
        return 1.00

def generate_ecg_data_and_animate():
    """Main function to generate ECG data and create animation"""
    
    print("Generating cumulative ECG data...")
    
    # Parameters
    fs = 250
    window_size = 30
    num_frames = 200
    plot_tail = 10
    duration_per_frame = 15
    
    # Storage
    all_data = []
    frame_count = 0
    cumulative_ecg_signal = []
    cumulative_time = []
    
    # Generate data frame by frame
    for i in range(num_frames):
        print(f"Processing Frame {frame_count}...")
        
        # Generate ECG segment with realistic heart rate variation
        base_hr = 70
        natural_variation = np.random.normal(0, 3)
        breathing_cycle = 0.15 * i
        breathing_effect = 3 * np.sin(2 * np.pi * breathing_cycle / 60)
        time_trend = 8 * np.sin(2 * np.pi * i / (num_frames * 0.6))
        random_walk = np.random.normal(0, 1) if i > 0 else 0
        
        current_hr = base_hr + natural_variation + breathing_effect + time_trend + random_walk
        current_hr = np.clip(current_hr, 55, 95)
        
        np.random.seed(i * 42 + 789)
        ecg_segment = nk.ecg_simulate(duration_per_frame, sampling_rate=fs, heart_rate=current_hr)
        np.random.seed(None)
        
        # Apply progressive noise
        base_noise_std = get_noise_std(frame_count)
        noise_variation_factor = np.random.uniform(0.8, 1.2)
        actual_noise_std = base_noise_std * noise_variation_factor
        
        gaussian_noise = np.random.normal(0, actual_noise_std, len(ecg_segment))
        
        if actual_noise_std > 0.1:
            hf_noise_factor = min(0.3, (actual_noise_std - 0.1) * 0.5)
            hf_noise = hf_noise_factor * np.random.normal(0, 1, len(ecg_segment))
            b_hf, a_hf = butter(4, 30, 'highpass', fs=fs)
            hf_noise = filtfilt(b_hf, a_hf, hf_noise)
        else:
            hf_noise = 0
        
        if actual_noise_std > 0.2:
            lf_drift_factor = min(0.2, (actual_noise_std - 0.2) * 0.3)
            lf_drift = lf_drift_factor * np.sin(2 * np.pi * 0.5 * np.arange(len(ecg_segment)) / fs)
        else:
            lf_drift = 0
        
        total_noise = gaussian_noise + hf_noise + lf_drift
        ecg_segment_noisy = ecg_segment + total_noise
        
        # Update cumulative ECG
        start_time_global = len(cumulative_ecg_signal) / fs
        cumulative_ecg_signal.extend(ecg_segment_noisy.tolist())
        segment_time = np.arange(start_time_global, start_time_global + duration_per_frame, 1/fs)[:len(ecg_segment_noisy)]
        cumulative_time.extend(segment_time.tolist())
        
        # Extract features from sliding window
        if len(cumulative_ecg_signal) >= window_size * fs:
            feature_window_size = window_size * fs
            ecg_for_features = np.array(cumulative_ecg_signal[-feature_window_size:])
            time_for_features = np.array(cumulative_time[-feature_window_size:])
            time_for_features = time_for_features - time_for_features[0]
            
            try:
                features = get_ecg_features(ecg_for_features, time_for_features, fs)
                features_valid = np.all(np.isfinite(features))
            except Exception as e:
                features = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
                features_valid = False
        else:
            features = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
            features_valid = False
        
        # Prepare animation data
        plot_window_size = plot_tail * fs
        if len(cumulative_ecg_signal) >= plot_window_size:
            ecg_plot = cumulative_ecg_signal[-plot_window_size:]
            time_plot = cumulative_time[-plot_window_size:]
            time_plot = [t - time_plot[0] for t in time_plot]
        else:
            ecg_plot = cumulative_ecg_signal.copy()
            time_plot = cumulative_time.copy()
            if len(time_plot) > 0:
                time_plot = [t - time_plot[0] for t in time_plot]
        
        # Store frame data
        row_data = {
            'frame': frame_count,
            'time_global': len(cumulative_ecg_signal) / fs,
            'noise_std': actual_noise_std,
            'base_noise_std': base_noise_std,
            'current_hr': current_hr,
            'ecg_plot': ecg_plot,
            'time_plot': time_plot,
            'buffer_full': len(cumulative_ecg_signal) >= window_size * fs,
            'cumulative_duration': len(cumulative_ecg_signal) / fs,
            'segment_duration': duration_per_frame
        }
        
        if features_valid:
            row_data.update({
                'hr_mean': features[0], 'hr_max': features[1], 'hr_min': features[2],
                'hrv': features[3], 'snr': features[4], 'features_valid': True
            })
        else:
            row_data.update({
                'hr_mean': np.nan, 'hr_max': np.nan, 'hr_min': np.nan,
                'hrv': np.nan, 'snr': np.nan, 'features_valid': False
            })

        all_data.append(row_data)
        frame_count += 1

    df = pd.DataFrame(all_data)
    
    # Compute predictions with rolling normalization
    valid_features = df[df['features_valid'] == True].copy()
    if len(valid_features) > 0:
        hr_mean_list = valid_features['hr_mean'].tolist()
        hr_max_list = valid_features['hr_max'].tolist()
        hr_min_list = valid_features['hr_min'].tolist()
        hrv_list = valid_features['hrv'].tolist()
        
        predictions = []
        for i in range(len(hr_mean_list)):
            current_hr_mean = hr_mean_list[:i+1]
            current_hr_max = hr_max_list[:i+1]
            current_hr_min = hr_min_list[:i+1]
            current_hrv = hrv_list[:i+1]
            
            current_data = np.column_stack([current_hr_mean, current_hr_max, current_hr_min, current_hrv])
            current_scaler = StandardScaler()
            current_data_std = current_scaler.fit_transform(current_data)
            current_features_std = current_data_std[-1]
            
            prediction = model(current_features_std.reshape(1, -1))
            
            if hasattr(prediction, 'numpy'):
                pred_val = float(prediction.numpy()[0][0])
            elif isinstance(prediction, dict):
                pred_val = float(next(iter(prediction.values()))[0][0])
            else:
                pred_val = float(prediction[0][0])
                
            predictions.append(pred_val)
        
        valid_features['prediction'] = predictions
    
    # Create animation
    print("Creating animation...")
    
    cumulative_features = {
        'frames': [], 'times': [], 'hr_mean': [], 'hr_max': [], 'hr_min': [], 
        'hrv': [], 'snr': [], 'predictions': [], 'pred_times': [],
        'ecg_data': [], 'ecg_times': []
    }
    
    def animate_from_dataframe(frame_idx):
        if frame_idx >= len(df):
            return
        
        row = df.iloc[frame_idx]
        
        for ax in axs:
            ax.clear()
        
        # Plot current ECG window
        if row['ecg_plot'] is not None and len(row['ecg_plot']) > 0:
            axs[0].plot(row['time_plot'], row['ecg_plot'], 'k-', linewidth=0.8)
            axs[0].set_title(f'Current ECG Window - Frame {row["frame"]} - Noise STD: {row["noise_std"]:.3f}')
            axs[0].set_ylabel('Amplitude (mV)')
            axs[0].grid(True, alpha=0.3)
        else:
            axs[0].set_title(f'ECG Signal - Frame {row["frame"]} - Buffer filling...')
            axs[0].set_ylabel('Amplitude (mV)')
            axs[0].grid(True, alpha=0.3)
        
        # Update cumulative features
        if row['features_valid'] and not pd.isna(row['hr_mean']):
            cumulative_features['frames'].append(row['frame'])
            cumulative_features['times'].append(row['time_global'])
            cumulative_features['hr_mean'].append(row['hr_mean'])
            cumulative_features['hr_max'].append(row['hr_max'])
            cumulative_features['hr_min'].append(row['hr_min'])
            cumulative_features['hrv'].append(row['hrv'])
            cumulative_features['snr'].append(row['snr'])
            
            if row['ecg_plot'] is not None and len(row['ecg_plot']) > 0:
                if len(cumulative_features['ecg_times']) == 0:
                    time_offset = 0
                    cumulative_features['ecg_times'].extend([time_offset + i/250 for i in range(len(row['ecg_plot']))])
                else:
                    last_time = cumulative_features['ecg_times'][-1]
                    time_step = 1/250
                    cumulative_features['ecg_times'].extend([last_time + time_step + i*time_step for i in range(len(row['ecg_plot']))])
                
                cumulative_features['ecg_data'].extend(row['ecg_plot'])
            
            # Add predictions
            if 'valid_features' in locals() and len(valid_features) > 0:
                matching_rows = valid_features[valid_features['frame'] == row['frame']]
                if len(matching_rows) > 0 and 'prediction' in matching_rows.columns:
                    prediction_val = matching_rows.iloc[0]['prediction']
                    if not pd.isna(prediction_val):
                        cumulative_features['predictions'].append(prediction_val)
                        cumulative_features['pred_times'].append(row['time_global'])
        
        # Plot cumulative ECG
        if len(cumulative_features['ecg_data']) > 0:
            axs[1].plot(cumulative_features['ecg_times'], cumulative_features['ecg_data'], 'k-', linewidth=0.8)
            
            if (row['features_valid'] and not pd.isna(row['hr_mean']) and 
                row['ecg_plot'] is not None and len(row['ecg_plot']) > 0):
                
                current_segment_length = len(row['ecg_plot'])
                if len(cumulative_features['ecg_times']) >= current_segment_length:
                    current_times = cumulative_features['ecg_times'][-current_segment_length:]
                    current_data = cumulative_features['ecg_data'][-current_segment_length:]
                    axs[1].plot(current_times, current_data, 'red', linewidth=1.2, alpha=0.8)
            
            total_duration = len(cumulative_features['ecg_data']) / 250
            axs[1].set_title(f'Cumulative ECG Signal - Total: {total_duration:.1f}s')
            axs[1].set_ylabel('Amplitude (mV)')
            axs[1].grid(True, alpha=0.3)
            
            if len(cumulative_features['ecg_times']) > 0:
                axs[1].set_xlim(cumulative_features['ecg_times'][0], cumulative_features['ecg_times'][-1])
        else:
            feature_count = len(cumulative_features['times'])
            axs[1].text(0.5, 0.5, f'Waiting for valid features to accumulate ECG...\n(Have {feature_count} valid frames)', 
                       ha='center', va='center', transform=axs[1].transAxes, fontsize=10)
            axs[1].set_title('Cumulative ECG Signal')
            axs[1].set_ylabel('Amplitude (mV)')
            axs[1].grid(True, alpha=0.3)
        
        # Plot feature time series
        if len(cumulative_features['times']) > 0:
            times = cumulative_features['times']
            
            axs[2].plot(times, cumulative_features['snr'], 'purple', linewidth=2)
            axs[2].set_title('Signal-to-Noise Ratio (SNR)')
            axs[2].set_ylabel('SNR (dB)')
            axs[2].set_ylim(0, 30)
            axs[2].grid(True, alpha=0.3)

            axs[3].plot(times, cumulative_features['hr_mean'], 'blue', linewidth=2)
            axs[3].set_title('Mean Heart Rate')
            axs[3].set_ylabel('HR (BPM)')
            axs[3].axhline(y=70, color='red', linestyle='--', alpha=0.5)
            axs[3].grid(True, alpha=0.3)

            axs[4].plot(times, cumulative_features['hr_max'], 'red', linewidth=2)
            axs[4].set_title('Maximum Heart Rate')
            axs[4].set_ylabel('HR (BPM)')
            axs[4].grid(True, alpha=0.3)

            axs[5].plot(times, cumulative_features['hr_min'], 'green', linewidth=2)
            axs[5].set_title('Minimum Heart Rate')
            axs[5].set_ylabel('HR (BPM)')
            axs[5].grid(True, alpha=0.3)

            axs[6].plot(times, cumulative_features['hrv'], 'black', linewidth=2)
            axs[6].set_title('Heart Rate Variability (RMSSD)')
            axs[6].set_ylabel('HRV (ms)')
            axs[6].grid(True, alpha=0.3)
        
        # Plot predictions
        if len(cumulative_features['predictions']) > 0:
            pred_times = cumulative_features['pred_times']
            predictions = cumulative_features['predictions']
            
            axs[7].plot(pred_times, predictions, 'black', linewidth=2)
            axs[7].axhline(y=0.5, linestyle='--', color='red', alpha=0.7)
            axs[7].fill_between(pred_times, predictions, 
                              where=(np.array(predictions) >= 0.5), 
                              color='red', alpha=0.3)
            axs[7].set_title(f'Model Prediction Probability ({len(predictions)} predictions)')
            axs[7].set_ylabel('Probability')
            axs[7].set_ylim([0, 1])
            axs[7].grid(True, alpha=0.3)
        else:
            feature_count = len(cumulative_features['times'])
            if feature_count > 0:
                axs[7].text(0.5, 0.5, f'Waiting for predictions...\n(Have {feature_count} features)', 
                           ha='center', va='center', transform=axs[7].transAxes, fontsize=10)
            else:
                axs[7].text(0.5, 0.5, 'Waiting for valid features...', 
                           ha='center', va='center', transform=axs[7].transAxes, fontsize=10)
            axs[7].set_title('Model Prediction Probability')
            axs[7].set_ylabel('Probability')
            axs[7].set_ylim([0, 1])
            axs[7].grid(True, alpha=0.3)
        
        axs[-1].set_xlabel('Time (s)')
        
        total_ecg_duration = len(cumulative_features['ecg_data']) / 250 if len(cumulative_features['ecg_data']) > 0 else 0
        total_features = len(cumulative_features['times'])
        total_predictions = len(cumulative_features['predictions'])
        
        fig.suptitle(f'Real-time ECG Analysis - Frame {frame_idx + 1}/{len(df)} | Cumulative ECG: {total_ecg_duration:.1f}s | Features: {total_features} | Predictions: {total_predictions}', fontsize=12)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
    
    fig, axs = plt.subplots(8, 1, figsize=(12, 12))
    
    anim = animation.FuncAnimation(
        fig, animate_from_dataframe, 
        frames=len(df), 
        interval=50,
        blit=False, 
        repeat=False
    )
    
    # Save animation
    try:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=25, metadata=dict(artist='ECG Analyzer'), bitrate=1800)
        anim.save('ecg_analysis_animation.mp4', writer=writer)
        print("Animation saved as 'ecg_analysis_animation.mp4'")
    except:
        try:
            anim.save('ecg_analysis_animation.gif', writer='pillow', fps=5)
            print("Animation saved as 'ecg_analysis_animation.gif'")
        except Exception as e:
            print(f"Could not save animation: {e}")
            plt.show()
    
    return df, valid_features

if __name__ == "__main__":
    df, valid_features = generate_ecg_data_and_animate()