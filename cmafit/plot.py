import csv
import numpy as np

class Plot:
    def __init__(self, filename):
        with open(filename, "rt") as inf:
            reader = csv.DictReader(inf)
            time = []
            voltage = []
            current = []
            for row in reader:
                time.append(float(row["time"]))
                voltage.append(float(row["voltage"]))
                current.append(float(row["current"]))
            self.time = np.array(time)
            self.voltage = np.array(voltage)
            self.current = np.array(current)
