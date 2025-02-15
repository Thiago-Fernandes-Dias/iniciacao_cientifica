from sklearn.metrics import accuracy_score, recall_score, precision_score

class MultiClassResults:
    y_true: list[str]
    y_pred: list[str]
    hp: dict[str, object]

    def __init__(self, y_true: list[str], y_pred: list[str], hp: dict[str, object]):
        self.y_true = y_true
        self.y_pred = y_pred
        self.hp = hp

    def accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)
    
    def recall(self):
        return recall_score(self.y_true, self.y_pred, average='micro')
    
    def precision(self):
        return precision_score(self.y_true, self.y_pred, average='micro')

    def to_dict(self)  -> dict[str, object]:
        return {
            "hp": self.hp,
            "accuracy": self.accuracy(),
            "recall": self.recall(),
            "precision": self.precision()
        }