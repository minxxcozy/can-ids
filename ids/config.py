# ids/config.py
# 컬럼 후보 alias 정의

TIMESTAMP_CANDIDATES = ["timestamp", "time", "ts"]
ID_CANDIDATES         = ["arbitration_id", "can_id", "id", "frameid"]
DATA_CANDIDATES       = ["data", "payload"]
LABEL_CANDIDATES      = ["class", "label", "attack", "type", "target"]
SUBLABEL_CANDIDATES   = ["subclass", "sub_label", "category"]

# Data_0 ~ Data_7 같은 바이트 컬럼 prefix
BYTE_PREFIXES = ["data_", "byte_"]