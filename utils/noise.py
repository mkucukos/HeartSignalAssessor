FLATLINE_START = 470
FLATLINE_END   = 500


def get_noise_std(frame_count):
    """Return the target noise STD for a given simulation frame."""
    schedule = [
        (15,  0.01),
        (35,  0.05),
        (55,  0.15),
        (80,  0.30),
        (100, 0.45),
        (130, 0.60),
        (160, 0.80),
        (200, 1.00),
        (280, 1.25),
        (330, 1.50),
        (370, 1.75),
        (410, 2.00),
        (430, 1.80),
        (450, 1.60),
        (470, 1.40),
    ]
    for threshold, std in schedule:
        if frame_count < threshold:
            return std
    return 1.40


def is_flatline_period(frame_count: int) -> bool:
    """Return True if this frame falls inside the simulated flatline window."""
    return FLATLINE_START <= frame_count < FLATLINE_END
