from sklearn.metrics import accuracy_score, recall_score, precision_score

class MultiClassResults:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)
    
    def recall(self):
        return recall_score(self.y_true, self.y_pred, average='micro')
    
    def precision(self):

        return precision_score(self.y_true, self.y_pred, average='micro')
    
    def print_results(self):
        print(f"- Accuracy: {self.accuracy()}")
        print(f"- Recall: {self.recall()}")
        print(f"- Precision: {self.precision()}")