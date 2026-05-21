# Flatline windows as (start_frame_inclusive, end_frame_exclusive)
# 10 s/frame: frames 100-120 = seconds 1000-1200, frames 200-220 = seconds 2000-2200
FLATLINE_WINDOWS = [(100, 120), (200, 220)]


def get_noise_std(frame_count):
    """Return the target noise STD for a given simulation frame."""
    schedule = [
        (10,  0.01),   # clean
        (25,  0.05),   # low noise
        (40,  0.15),   # moderate
        (60,  0.30),   # high
        (80,  0.50),   # extreme  → flatline 1 at frame 100
        (140, 1.00),   # post-flatline-1 escalation
        (160, 1.25),
        (180, 1.50),
        (200, 2.00),   # peak     → flatline 2 at frame 200
        (240, 1.50),   # recovery
        (260, 1.00),
        (280, 0.50),
        (300, 0.10),
    ]
    for threshold, std in schedule:
        if frame_count < threshold:
            return std
    return 0.10


def is_flatline_period(frame_count: int) -> bool:
    """Return True if this frame falls inside any simulated flatline window."""
    return any(start <= frame_count < end for start, end in FLATLINE_WINDOWS)
