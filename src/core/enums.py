from enum import Enum


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"
    UPLOADED = "uploaded"


class Resolution(Enum):
    RES_480P = "480p"
    RES_720P = "720p"
    RES_1080P = "1080p"


class AspectRatio(Enum):
    AR_16_9 = "16:9 Landscape"
    AR_9_16 = "9:16 Portrait"
    AR_1_1 = "1:1 Square"


class Duration(Enum):
    DUR_5S = "5 Seconds (121 frames)"
    DUR_10S = "10 Seconds (241 frames)"
    DUR_15S = "15 Seconds (361 frames)"
