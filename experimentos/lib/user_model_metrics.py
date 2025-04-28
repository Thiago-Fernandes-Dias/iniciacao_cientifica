class UserModelMetrics:
    def __init__(self, user_id: str, far: float, frr: float):
        self.user_id = user_id
        self.frr = frr
        self.far = far

    def to_dict(self):
        return {
            "user_id": self.user_id,
            "far": self.far,
            "frr": self.frr
        }

    @staticmethod
    def from_dict(data: dict):
        return UserModelMetrics(
            user_id=data["user_id"],
            frr=data["frr"],
            far=data["far"]
        )