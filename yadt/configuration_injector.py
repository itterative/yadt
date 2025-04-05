from injector import Module, singleton, provider

from yadt.configuration import Configuration
from yadt.db_dataset import DatasetDB
from yadt.tagger_shared import Predictor

class InjectorConfiguration(Module, Configuration):
    def configure(self, binder):
        binder.bind(Configuration, to=Configuration(
            device=self.device,
            cache_folder=self.cache_folder,
            score_character_threshold=self.score_character_threshold,
            score_general_threshold=self.score_general_threshold,
            score_slider_step=self.score_slider_step
        ))

    @singleton
    @provider
    def provide_dataset_db(self, configuration: Configuration) -> DatasetDB:
        return DatasetDB(configuration)

    @singleton
    @provider
    def provide_preditor(self) -> Predictor:
        return Predictor()
