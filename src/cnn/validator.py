"""
Author: Jakub Lisowski, Tomasz Mycielski, 2024

Simple class providing a simple validation method
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate
from torch import Tensor
from src.constants import NUM_CLASSES_UNKNOWN

class Validator:
    """
    Class providing a simple validation

    Class keeps track of predictions, then calculates macro-F1 and accuracy scores
    """

    _results: pd.DataFrame
    le: LabelEncoder

    def __init__(self, le: LabelEncoder = None) -> None:
        """
        Method initializing the validation
        """
        if le is None:
            classes = [0, 1]
        else:
            classes = le.classes_

        self.le = le
        self._results = pd.DataFrame(0, columns=classes, index=classes, dtype='int64')

    def _flattened_(self) -> pd.DataFrame:
        res = self._results
        # columns
        unknowns = sum(res[f'unknown{i}'] for i in range(1, NUM_CLASSES_UNKNOWN + 1))
        res = res.assign(unknown = unknowns)
        for i in [f'unknown{i}' for i in range(1, NUM_CLASSES_UNKNOWN + 1)]:
            del res[i]
        # rows
        unknowns = sum(res.loc[f'unknown{i}'] for i in range(1, NUM_CLASSES_UNKNOWN + 1))
        res.loc['unknown'] = unknowns
        res = res.drop(
            index = [f'unknown{i}' for i in range(1, NUM_CLASSES_UNKNOWN + 1)])
        return res

    def validate(self, predictions: Tensor, target: Tensor) -> None:
        """
        Method saving the results of the validation
        """
        if self.le is None:
            for response, answer in zip(predictions, target):
                self._results.loc[answer.item(), response.argmax(0).item()] += 1
        else:
            for response, answer in zip(predictions, target):
                loc = tuple(self.le.inverse_transform((answer.item(), response.argmax(0).item())))
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
        res = self._results if 'unknown2' not in list(self._results.columns.values) else self._flattened_()
        macro_f1 = 0
        for i in res:
            numerator = 2 * res[i][i]
            # fn + fp + 2tp of a class is the sum of its row + sum of its column
            denominator = res.sum(axis=0)[i] + res.sum(axis=1)[i]
            macro_f1 += numerator / denominator
        return macro_f1 / len(res)

    def get_accuracy(self) -> float:
        """
        Method calculating overall accuracy
        """
        res = self._results if 'unknown2' not in list(self._results.columns.values) else self._flattened_()
        return np.diag(res).sum() / res.values.sum()

    def get_results_str(self) -> str:
        """
        Method returning the results as a string
        """

        f1 = self.get_f1_score()
        if not f1:
            f1 = 'N/A'
        res = self._results if 'unknown2' not in list(self._results.columns.values) else self._flattened_()
        table = tabulate(res,
                         ['Pred. ' + str(i) for i in res.columns],
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
