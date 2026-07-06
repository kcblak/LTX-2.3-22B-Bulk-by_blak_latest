from enum import Enum, auto


class JobStatus(Enum):
    PENDING = auto()
    VALIDATING = auto()
    READY = auto()
    LOADING = auto()
    GENERATING = auto()
    ENCODING = auto()
    VERIFYING = auto()
    VERIFIED = auto()
    UPLOAD_PENDING = auto()
    UPLOADING = auto()
    VERIFYING_REMOTE = auto()
    UPLOADED = auto()
    COMPLETED = auto()
    FAILED_VALIDATION = auto()
    FAILED_RENDER = auto()
    FAILED_UPLOAD = auto()
    FAILED_VERIFY = auto()
    INTERRUPTED = auto()
    RETRYING = auto()


class Duration(Enum):
    D5 = ("5 Seconds", 5, 121)
    D10 = ("10 Seconds", 10, 241)
    D15 = ("15 Seconds", 15, 361)

    def __init__(self, label, seconds, frames):
        self.label = label
        self.seconds = seconds
        self.frames = frames

    @classmethod
    def from_string(cls, s: str) -> "Duration":
        s_lower = s.lower()
        for d in cls:
            if d.label.lower() in s_lower:
                return d
        raise ValueError(f"Invalid duration string: {s}")


class Resolution(Enum):
    R480 = ("480p", 480)
    R720 = ("720p", 720)
    R1080 = ("1080p", 1080)

    def __init__(self, label, pixels):
        self.label = label
        self.pixels = pixels

    @classmethod
    def from_string(cls, s: str) -> "Resolution":
        s_lower = s.lower()
        for r in cls:
            if r.label.lower() in s_lower:
                return r
        raise ValueError(f"Invalid resolution string: {s}")


class AspectRatio(Enum):
    AR_16_9 = ("16:9 Landscape", 16, 9)
    AR_9_16 = ("9:16 Portrait", 9, 16)
    AR_1_1 = ("1:1 Square", 1, 1)

    def __init__(self, label, width_ratio, height_ratio):
        self.label = label
        self.width_ratio = width_ratio
        self.height_ratio = height_ratio

    @classmethod
    def from_string(cls, s: str) -> "AspectRatio":
        s_lower = s.lower()
        for ar in cls:
            if ar.label.lower() in s_lower:
                return ar
        raise ValueError(f"Invalid aspect ratio string: {s}")
