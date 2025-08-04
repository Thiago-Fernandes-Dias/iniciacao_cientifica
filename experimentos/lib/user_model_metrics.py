class UserModelMetrics:
    frr: float
    far: float

    def __init__(self, frr: float, far: float):
        self.frr = frr
        self.far = far