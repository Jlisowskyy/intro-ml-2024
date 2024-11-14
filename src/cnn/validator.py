"""
Author: Jakub Lisowski, Tomasz Mycielski, 2024

Simple class providing a simple validation method
"""
from torch import Tensor


class Validator:
    """
    Class providing a simple validation

    Class performs:
    - counting the number of true positives, false positives, false negatives, true negatives
    - calculating the accuracy, F1 score, macro F1 score
    """

    _results: list[list[int]]

    def __init__(self) -> None:
        """
        Method initializing the validation
        """

        self._results = [[0, 0], [0, 0]]

    def get_results(self) -> list[list[int]]:
        """
        Method returning the results
        """

        return self._results

    def validate(self, predictions: Tensor, target: Tensor) -> None:
        """
        Method saving the results of the validation
        """

        for response, answer in zip(predictions, target):
            self._results[response.argmax(0).item()][answer.item()] += 1

    def get_results_str(self) -> str:
        """
        Method returning the results as a string
        """

        f1 = 2 * self._results[1][1] / (
                2 * self._results[1][1] + self._results[0][1] + self._results[1][0])

        # macro F1 score which assumes that both classes are positive
        macro_f1 = (self._results[1][1] / (
                2 * self._results[1][1] + self._results[0][1] + self._results[1][0]) +
                    self._results[0][0] / (
                            2 * self._results[0][0] + self._results[0][1] + self._results[1][0]))
        accuracy = (self._results[1][1] + self._results[0][0]) / (
                sum(self._results[0]) + sum(self._results[1]))

        return f'''
┏━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃      ┃ Pred. Pos. ┃ Pred. Neg. ┃
┣━━━━━━╋━━━━━━━━━━━━╋━━━━━━━━━━━━┫
┃ Pos. ┃ {self._results[1][1]:>10} ┃ {self._results[0][1]:>10} ┃
┣━━━━━━╋━━━━━━━━━━━━╋━━━━━━━━━━━━┫
┃ Neg. ┃ {self._results[1][0]:>10} ┃ {self._results[0][0]:>10} ┃
┗━━━━━━┻━━━━━━━━━━━━┻━━━━━━━━━━━━┛

Accuracy: {accuracy}
F1 score: {f1}
Macro F1: {macro_f1}
'''

    def display_results(self) -> None:
        """
        Method displaying the results
        """

        print(self.get_results_str())

    def __add__(self, b : 'Validator') -> 'Validator':
        out = Validator()
        out._results[0][0] = self._results[0][0] + b.get_results()[0][0]
        out._results[0][1] = self._results[0][1] + b.get_results()[0][1]
        out._results[1][0] = self._results[1][0] + b.get_results()[1][0]
        out._results[1][1] = self._results[1][1] + b.get_results()[1][1]
        return out
