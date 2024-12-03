from abc import ABC, abstractmethod


class GeneralEvaluator(ABC):

    def __init__(self, metric_name, gts_path=None, verbose=True):
        self.verbose = verbose  # print progress bar and results or not
        self.metric_name = metric_name
        self.ground_truths = None
        # if gts_path is not None, load ground truth files from disk, set it manually otherwise.
        if gts_path is not None:
            self._set_ground_truths_from_files(gts_path)

    def set_ground_truths(self, ground_truths):
        self.ground_truths = ground_truths


    @abstractmethod
    def _print_results(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, predictions):
        pass

