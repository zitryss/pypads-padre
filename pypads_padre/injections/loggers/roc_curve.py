from typing import Type, Optional

from pydantic import BaseModel
from pypads.app.env import InjectionLoggerEnv
from pypads.app.injections.injection import InjectionLogger
from pypads.app.injections.tracked_object import TrackedObject
from pypads.model.logger_output import OutputModel, TrackedObjectModel
from pypads.model.models import IdReference
from pypads.utils.logging_util import FileFormats


class RocTO(TrackedObject):
    class RocTOModel(TrackedObjectModel):
        value: str = ...

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

        roc_to = RocTO(parent=_logger_output)
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
