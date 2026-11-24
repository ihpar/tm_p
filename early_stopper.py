class EarlyStopper:
    def __init__(self, patience=1):
        self.patience = patience
        self.counter = 0
        self.min_validation_loss = float("inf")

    def is_improvement(self, validation_loss):
        """
        Returns True if the validation loss is lower than every loss seen
        so far. Otherwise counts one more epoch without improvement.
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            return True

        self.counter += 1
        return False

    def early_stop(self):
        """
        Returns True after `patience` epochs in a row without improvement.
        """
        return self.counter >= self.patience
