
class Record:
    def __init__(self, ratio, device_type="Jetson NX",):
        self.ratio = ratio
        self.device = []
        self.device_type = device_type
        self.memory = {}
        self.latency = {}
        
    def tostring(self, data, name):
        data_str = ""
        for i, d in enumerate(data):
            data_str += f"{name}{i+1}: {d} "
        return data_str
            
    
    def __str__(self):
        record_str = f"Model: {self.ratio} Device Type: {self.device_type} Device: {self.device}"
        # record_str += self.tostring(self.memory, "mem")
        # record_str += self.tostring(self.latency, "lat")
        return record_str
        