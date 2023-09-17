import numpy as np
import torch

class BestSaver:
    def __init__(self, model, save_path, best_criterion='min'):
        self.model = model
        self.save_path = save_path
        self.best_criterion = best_criterion

        if best_criterion == 'min':
            self.__record_criterion = np.inf
        elif best_criterion == 'max':
            self.__record_criterion = -np.inf
        else:
            raise NameError("best_criterion must be \'max\' or \'min\'")

    def saveJudgment(self, criterion_value):
        if self.best_criterion == 'min':
            if criterion_value < self.__record_criterion:
                self.__record_criterion = criterion_value
                torch.save(self.model.state_dict(), self.save_path)

        elif self.best_criterion == 'max':
            if criterion_value > self.__record_criterion:
                self.__record_criterion = criterion_value
                torch.save(self.model.state_dict(), self.save_path)