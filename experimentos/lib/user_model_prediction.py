class UserModelPrediction:
    user_id: str
    session: int
    repetition: int
    predicted: int
    expected: int

    def __init__(self, user_id: str, expected: int, predicted: int, session: int, repetition: int):
        self.user_id = user_id
        self.predicted = predicted
        self.expected = expected
        self.session = session
        self.repetition = repetition

    def __repr__(self):
        return f"UserModelPrediction(user_id={self.user_id}, predicted={self.predicted})"
    
    def to_dict(self) -> dict[str, object]:
        return {
            "user_id": self.user_id,
            "expected": self.expected,
            "predicted": self.predicted,
            "session": self.session,
            "repetition": self.repetition
        }
    
    @staticmethod
    def from_dict(data: dict[str, object]) -> "UserModelPrediction":
        return UserModelPrediction(
            user_id=data["user_id"],
            expected=data["expected"],
            predicted=data["predicted"],
            session=data["session"],
            repetition=data["repetition"]
        )
