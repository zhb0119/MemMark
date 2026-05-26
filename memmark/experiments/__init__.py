"""RQ runners (README §10.3–§10.7).

Each module is a thin function over the shared driver output and the
audit trace, so you can mix-and-match:

  driver_result = LoCoMoDriver(...).run(conversation)
  rq1 = run_rq1_utility(driver_result, baseline_results)
  rq2 = run_rq2_capacity(driver_result)
  rq3 = run_rq3_in_record(driver_result, secret_key=...)
  rq4 = run_rq4_robustness(driver_result, secret_key=...)
  rq5 = run_rq5_integrity(driver_result, ground_truth_qa=...)
"""

from memmark.experiments.rq1_utility import run_rq1_utility
from memmark.experiments.rq2_capacity import run_rq2_capacity
from memmark.experiments.rq3_in_record import run_rq3_in_record
from memmark.experiments.rq4_robustness import run_rq4_robustness
from memmark.experiments.rq5_integrity import run_rq5_integrity

__all__ = [
    "run_rq1_utility",
    "run_rq2_capacity",
    "run_rq3_in_record",
    "run_rq4_robustness",
    "run_rq5_integrity",
]
