class UserModelMetrics:
    frr: float
    far: float

    def __init__(self, frr: float, far: float):
        self.frr = frr
        self.far = far
    
    def getBAcc(self) -> float:
        return 1 - (self.frr + self.far) / 2 if (self.frr + self.far) > 0 else 1.0