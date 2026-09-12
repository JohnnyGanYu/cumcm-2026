"""Keep adaptive coverage stations stable until physically visited or redundant."""
from src.b_q3_combined_experiment import FastClear,FastPig


class StableMixin:
    def _cover_plan(self):
        pending=[st for st in self.channels.values() if self._search_target(st)]
        if not pending:return []
        remaining=[p for p in getattr(self,'_scheduled_cover',[])
                   if any(st.feasible.coverage_gain(*p)>0 for st in pending)]
        if not remaining:
            self._cover_cache=None
            remaining=super()._cover_plan()
        self._scheduled_cover=list(remaining)
        return list(remaining)

    def _visit_cover_station(self,station):
        self._scheduled_cover=[p for p in getattr(self,'_scheduled_cover',[]) if p!=station]
        return super()._visit_cover_station(station)


class StableFast(StableMixin,FastClear):
    pass


class StablePig(StableMixin,FastPig):
    pass


class DeferredFast(FastPig):
    def __init__(self,client,config=None):
        super().__init__(client,config)
        self.config.cover_defer=True
