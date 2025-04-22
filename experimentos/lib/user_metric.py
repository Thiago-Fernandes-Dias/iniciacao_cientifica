class UserModelMetric:
    def __init__(self, user_id: str, value: float):
        self.user_id = user_id
        self.value = value

    def to_dict(self):
        return {
            "user_id": self.user_id,
            "value": self.value
        }

    @staticmethod
    def from_dict(data: dict):
        return UserModelMetric(
            user_id=data["user_id"],
            value=data["value"]
        )