from typing import Type, Optional, List

import numpy as np
from pydantic import BaseModel
from pypads.app.env import InjectionLoggerEnv
from pypads.app.injections.injection import InjectionLogger
from pypads.app.injections.tracked_object import TrackedObject
from pypads.model.logger_output import OutputModel, TrackedObjectModel
from pypads.model.models import IdReference
from pypads.utils.logging_util import FileFormats
from sklearn.metrics import roc_curve, auc


class RocTO(TrackedObject):
    class RocTOModel(TrackedObjectModel):
        category: str = "ROC category"
        name: str = "ROC name"
        description: str = "ROC description"
        tpr: List[float] = []
        fpr: List[float] = []

    @classmethod
    def get_model_cls(cls) -> Type[BaseModel]:
        return cls.RocTOModel


class RocILF(InjectionLogger):
    class RocILFOutput(OutputModel):  # Is it supposed to be a nested class?
        roc: Optional[IdReference] = None

        class Config:
            orm_mode = True

    @classmethod
    def output_schema_class(cls) -> Type[OutputModel]:
        return cls.RocILFOutput

    def __post__(self, ctx, *args, _pypads_env: InjectionLoggerEnv,
                 _pypads_artifact_fallback: Optional[FileFormats] = None, _logger_call,
                 _logger_output, _pypads_result,
                 **kwargs):  # Why the order of parameters changes from logger to logger?
        # pr = _pypads_result  # holds returned value of the function being tracked
        # lo = _logger_output  # RocILFOutput initialized with the ref to RocTO

        from pypads.app.pypads import get_current_pads
        pads = get_current_pads()

        preds = _pypads_result
        if pads.cache.run_exists("predictions"):
            preds = pads.cache.run_pop("predictions")

        # check if there is info about decision scores
        probabilities = None
        if pads.cache.run_exists("probabilities"):
            probabilities = pads.cache.run_pop("probabilities")

        # check if there is info on truth values
        targets = None  # labels
        if pads.cache.run_exists("targets"):
            targets = pads.cache.run_get("targets")


        roc_to = RocTO(parent=_logger_output)

        new_preds = []
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(probabilities)):  # || len(preds)
            fpr[i], tpr[i], _ = roc_curve(preds[:, i], probabilities[i][:, 1])
            new_preds.append(probabilities[i][:, 1])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(np.ravel(preds), np.ravel(new_preds))
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        roc_to.tpr = tpr["micro"].tolist()
        roc_to.fpr = fpr["micro"].tolist()

        roc_to.value = "hello, world!"

        _logger_output.roc = roc_to.store()



        # Many splits
        # decisions = RocTO(split_id=uuid.UUID(split_id), parent=_logger_output)
        # _logger_output.individual_decisions.append(decisions.store())
        # store() function returns a reference to where it was stored

        # One split
        # decisions = RocTO(split_id=split_id, parent=_logger_output)
        # _logger_output.individual_decisions = decisions.store()

        # probabilities, predictions, truth_values
