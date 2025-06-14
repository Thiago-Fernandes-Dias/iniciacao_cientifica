class UserModelPrediction:
    user_id: str
    session: int
    repetition: int
    prediction: int
    expected: int

    def __init__(self, user_id: str, model_name: str, expected: int, prediction: int, session: int, repetition: int):
        self.user_id = user_id
        self.prediction = prediction
        self.expected = expected
        self.session = session
        self.repetition = repetition

    def __repr__(self):
        return f"UserModelPrediction(user_id={self.user_id}, model_name={self.model_name}, prediction={self.prediction})"
    
    def to_dict(self) -> dict[str, object]:
        return {
            "user_id": self.user_id,
            "expected": self.expected,
            "prediction": self.prediction,
            "session": self.session,
            "repetition": self.repetition
        }
    
    @staticmethod
    def from_dict(data: dict[str, object]) -> "UserModelPrediction":
        return UserModelPrediction(
            user_id=data["user_id"],
            expected=data["expected"],
            prediction=data["prediction"],
            session=data["session"],
            repetition=data["repetition"]
        )
