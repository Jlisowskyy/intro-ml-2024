"""
Author: Jakub Lisowski, Tomasz Mycielski, 2024

Simple class providing a simple validation method
"""
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate
from torch import Tensor


class Validator:
    """
    Class providing a simple validation

    Class keeps track of predictions, then calculates macro-F1 and accuracy scores
    """

    _results: pd.DataFrame

    def __init__(self, classes: Iterable[any] = None) -> None:
        """
        Method initializing the validation
        """
        if not classes:
            classes = [0, 1]

        self._results = pd.DataFrame(0, columns=classes, index=classes, dtype='int64')

    def validate(self, predictions: Tensor, target: Tensor,
                 le: LabelEncoder = None) -> None:
        """
        Method saving the results of the validation
        """
        if not le:
            for response, answer in zip(predictions, target):
                self._results.loc[answer.item(), response.argmax(0).item()] += 1
        else:
            for response, answer in zip(predictions, target):
                loc = tuple(le.inverse_transform((answer.item(), response.argmax(0).item())))
                self._results.loc[loc] += 1

    def get_f1_score(self) -> float | None:
        """
        Method returning F1 score if the validator classes are `[negative, positive]`
        """
        if (len(self._results.columns) == 2 and
                (self._results.columns == [0, 1]).all()):
            return 2 * self._results[1][1] / (
                    2 * self._results[1][1] + self._results[0][1] + self._results[1][0])
        return None

    def get_macro_f1(self) -> float:
        """
        Method calculating macro F1 score
        """
        macro_f1 = 0
        for i in self._results:
            numerator = 2 * self._results[i][i]
            # fn + fp + 2tp of a class is the sum of its row + sum of its column
            denominator = self._results.sum(axis=0)[i] + self._results.sum(axis=1)[i]
            macro_f1 += numerator / denominator
        return macro_f1 / len(self._results)

    def get_accuracy(self) -> float:
        """
        Method calculating overall accuracy
        """
        return np.diag(self._results).sum() / self._results.values.sum()

    def get_results_str(self) -> str:
        """
        Method returning the results as a string
        """

        f1 = self.get_f1_score()
        if not f1:
            f1 = 'N/A'

        table = tabulate(self._results,
                         ['Pred. ' + str(i) for i in self._results.columns],
                         tablefmt='heavy_grid')
        return f'''{table}

Accuracy: {self.get_accuracy()}
F1 score: {f1}
Macro F1: {self.get_macro_f1()}
'''

    def display_results(self) -> None:
        """
        Method displaying the results
        """

        print(self.get_results_str())

    def __add__(self, b: 'Validator') -> 'Validator':
        out = Validator()
        out._results = self._results + b._results
        return out
