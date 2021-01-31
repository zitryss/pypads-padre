import uuid
from typing import Type, Optional, Union, List

from pydantic import BaseModel
from pypads.app.env import InjectionLoggerEnv
from pypads.app.injections.injection import InjectionLogger
from pypads.app.injections.tracked_object import TrackedObject
from pypads.model.logger_output import OutputModel, TrackedObjectModel
from pypads.model.models import IdReference, BaseStorageModel
from pypads.utils.logging_util import FileFormats


class RocTO(TrackedObject):
    class RocTOModel(TrackedObjectModel):
        class DecisionModel(BaseStorageModel):
            truth: Union[str, int] = None
            prediction: Union[str, int] = ...

            # class Config:
            #     orm_mode = True

        decisions: List[DecisionModel] = []

    @classmethod
    def get_model_cls(cls) -> Type[BaseModel]:
        return cls.RocTOModel


class RocILF(InjectionLogger):
    # _needed_cached = SystemStatsTO.__name__

    class RocILFOutput(OutputModel):
        individual_decisions: Union[List[IdReference], IdReference] = None

        # class Config:
        #     orm_mode = True

    @classmethod
    def output_schema_class(cls) -> Type[OutputModel]:
        return cls.RocILFOutput

    def __post__(self, ctx, *args, _pypads_env: InjectionLoggerEnv,
                 _pypads_artifact_fallback: Optional[FileFormats] = None, _logger_call,
                 _logger_output, _pypads_result, **kwargs):
        pr = _pypads_result  # holds returned value
        lc = _logger_call  # RocILFOutput initialized with the ref to RocTO
        # decisions = RocTO(split_id=uuid.UUID(split_id), parent=_logger_output)
        # _logger_output.individual_decisions = decisions.store()
