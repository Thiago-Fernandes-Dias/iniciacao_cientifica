class UserModelPrediction:
    user_id: str
    session: int
    repetition: int
    model_name: str
    prediction: int
    expected: int

    def __init__(self, user_id: str, model_name: str, expected: int, prediction: int, session: int, repetition: int):
        self.user_id = user_id
        self.model_name = model_name
        self.prediction = prediction
        self.expected = expected
        self.session = 0
        self.repetition = 0

    def __repr__(self):
        return f"UserModelPrediction(user_id={self.user_id}, model_name={self.model_name}, prediction={self.prediction})"'