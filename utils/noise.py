# Flatline windows as (start_frame_inclusive, end_frame_exclusive)
# 10 s/frame: frames 100-120 = seconds 1000-1200, frames 200-220 = seconds 2000-2200
FLATLINE_WINDOWS = [(100, 120), (200, 220)]


def get_noise_std(frame_count):
    """Return the target noise STD for a given simulation frame."""
    schedule = [
        (20,  0.01),   # clean
        (35,  0.03),
        (50,  0.07),
        (65,  0.12),
        (80,  0.20),
        (95,  0.35),   # → flatline 1 at frame 100
        (130, 0.50),   # post-flatline-1
        (145, 0.70),
        (160, 0.90),
        (175, 1.20),
        (190, 1.60),
        (200, 2.00),   # peak → flatline 2 at frame 200
        (230, 1.60),   # recovery
        (250, 1.10),
        (265, 0.70),
        (280, 0.30),
        (300, 0.10),
    ]
    for threshold, std in schedule:
        if frame_count < threshold:
            return std
    return 0.10


def is_flatline_period(frame_count: int) -> bool:
    """Return True if this frame falls inside any simulated flatline window."""
    return any(start <= frame_count < end for start, end in FLATLINE_WINDOWS)
