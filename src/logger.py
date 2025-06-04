# src/logger.py

import csv
import os


class CSVLogger:
    """
    Логирует метрики в CSV (epoch, reward, free_energy и т.д.)
    """

    def __init__(self, filepath, header):
        self.filepath = filepath
        self.header = header
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    def log(self, values):
        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(values)
