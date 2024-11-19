"""
Author: Jakub Lisowski, Tomasz Mycielski, 2024

Simple class providing a simple validation method
"""
from collections.abc import Iterable, Sequence

import pandas as pd
from tabulate import tabulate
from torch import Tensor

class Validator:
    """
    Class providing a simple validation

    Class performs:
    - counting the number of true positives, false positives, false negatives, true negatives
    - calculating the accuracy, F1 score, macro F1 score
    """

    _results: pd.DataFrame

    def __init__(self, classes: Iterable[any]=None) -> None:
        """
        Method initializing the validation
        """
        if not classes:
            classes = [0, 1]

        self._results = pd.DataFrame(0, columns=classes, index=classes, dtype='int64')


    def validate(self, predictions: Tensor, target: Tensor,
                 mapping: Sequence[int]=None) -> None:
        """
        Method saving the results of the validation
        """
        if not mapping:
            for response, answer in zip(predictions, target):
                self._results.loc[answer.item(), response.argmax(0).item()] += 1
        else:
            for response, answer in zip(predictions, target):
                self._results[mapping[response.argmax(0).item()]][mapping[answer.item()]] += 1

    def get_results_str(self) -> str:
        """
        Method returning the results as a string
        """

        # F1 is only relvant when there's a nega
        f1 = 'N/A'
        if (len(self._results.columns) == 2 and
            (self._results.columns == [0, 1]).all()):
            f1 = 2 * self._results[1][1] / (
                    2 * self._results[1][1] + self._results[0][1] + self._results[1][0])

        # macro F1 score which assumes that both classes are positive
        macro_f1 = 0
        for i in self._results:
            numerator = 2 * self._results[i][i]
            # fn + fp + 2tp of a class is the sum of its row + sum of its column
            denominator = self._results.sum(axis=0)[i] + self._results.sum(axis=1)[i]
            macro_f1 += numerator/denominator
        macro_f1 /= len(self._results)
        accuracy = (self._results[1][1] + self._results[0][0]) / (
                sum(self._results[0]) + sum(self._results[1]))
        table = tabulate(self._results,
                         ['Pred. ' + str(i) for i in self._results.columns],
                         tablefmt='heavy_grid')
        return f'''{table}

Accuracy: {accuracy}
F1 score: {f1}
Macro F1: {macro_f1}
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
