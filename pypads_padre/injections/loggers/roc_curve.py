from typing import Type, Optional, List

import numpy as np
from pydantic import BaseModel
from pypads.app.env import InjectionLoggerEnv
from pypads.app.injections.injection import InjectionLogger
from pypads.app.injections.tracked_object import TrackedObject
from pypads.app.pypads import get_current_pads
from pypads.model.logger_output import OutputModel, TrackedObjectModel
from pypads.model.models import IdReference
from pypads.utils.logging_util import FileFormats

from pypads_padre.concepts.util import _len


class RocTO(TrackedObject):
    class RocTOModel(TrackedObjectModel):
        category: str = "ROC category"
        name: str = "ROC name"
        description: str = "ROC description"
        tpr: List[float] = []
        fpr: List[float] = []
        chart: str = ""

    @classmethod
    def get_model_cls(cls) -> Type[BaseModel]:
        return cls.RocTOModel


class RocILF(InjectionLogger):
    _dependencies = {"sklearn", "pandas", "altair"}

    class RocILFOutput(OutputModel):
        roc: Optional[IdReference] = None

        class Config:
            orm_mode = True

    @classmethod
    def output_schema_class(cls) -> Type[OutputModel]:
        return cls.RocILFOutput

    def __post__(self, ctx, *args, _pypads_env: InjectionLoggerEnv,
                 _pypads_artifact_fallback: Optional[FileFormats] = None, _logger_call,
                 _logger_output, _pypads_result,
                 **kwargs):

        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve
        import pandas as pd
        import altair as alt

        pads = get_current_pads()

        preds = _pypads_result
        if pads.cache.run_exists("predictions"):
            preds = pads.cache.run_get("predictions")

        targets = None
        if pads.cache.run_exists("targets"):
            targets = pads.cache.run_get("targets")
        if targets is None:
            return

        classes = np.unique(targets)

        mode = None
        current_split = None
        if pads.cache.run_exists("current_split"):
            mode = pads.cache.get("tracking_mode", "single")
            split_id = pads.cache.run_get("current_split")
            splitter = pads.cache.run_get(pads.cache.run_get("split_tracker"))
            splits = splitter.get("output").splits.splits
            current_split = splits.get(str(split_id), None)
        if mode == "multiple" and _len(preds) == _len(targets):
            return
        if current_split is None:
            return
        if current_split.test_set is None:
            return

        truth = []
        for instance in current_split.test_set:
            truth.append(targets[instance])
        if len(truth) == 0:
            return

        truth = label_binarize(truth, classes=classes)

        probabilities = None
        if pads.cache.run_exists("probabilities"):
            probabilities = pads.cache.run_get("probabilities")
        if probabilities is None:
            return

        fpr_mean = [0 for _ in range(len(classes))]
        tpr_mean = [0 for _ in range(len(classes))]
        for i in range(len(classes)):
            fpr, tpr, thresholds = roc_curve(truth[:, i], probabilities[:, i])
            fpr_mean = [f + fpr[j] / len(classes) for j, f in enumerate(fpr_mean)]
            tpr_mean = [t + tpr[j] / len(classes) for j, t in enumerate(tpr_mean)]

        roc_df = pd.DataFrame()
        roc_df['fpr'] = fpr_mean
        roc_df['tpr'] = tpr_mean
        chart = alt.Chart(roc_df).mark_line(color='red').encode(
            alt.X('fpr', title="false positive rate"),
            alt.Y('tpr', title="true positive rate"))

        roc_to = RocTO(parent=_logger_output)
        roc_to.fpr = fpr_mean
        roc_to.tpr = tpr_mean
        roc_to.chart = chart.to_json()
        _logger_output.roc = roc_to.store()
