"""Expanded development comparison before final Q3 configuration freeze."""
from src.b_q3_combined_experiment import FastClear
from src.b_q3_refine_experiment import RefineStrategy


class Candidate(FastClear):
    fraction=.85
    lateral=.04


class AllBearings(Candidate):
    def __init__(self,client,config=None):
        super().__init__(client,config)
        self.config.piggyback_max_per_stop=20


class SelectedPig(AllBearings):
    def __init__(self,client,config=None):
        super().__init__(client,config)
        self.config.piggyback_coverage=True
        self.config.piggyback_min_gain_cells=150


class SelectedRefine(Candidate,RefineStrategy):
    pass
