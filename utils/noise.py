def get_noise_std(frame_count):
    """Return the target noise STD for a given simulation frame."""
    schedule = [
        (20,  0.01),
        (50,  0.05),
        (75,  0.15),
        (100, 0.30),
        (125, 0.45),
        (150, 0.60),
        (175, 0.80),
        (200, 1.00),
        (450, 1.25),
        (500, 1.50),
        (550, 1.75),
        (600, 2.00),
        (650, 1.80),
        (700, 1.60),
        (750, 1.40),
        (800, 1.20),
        (850, 1.00),
    ]
    for threshold, std in schedule:
        if frame_count < threshold:
            return std
    return 1.00
