"""Coverage Intelligence Platform - reference implementation of the CF-SCD algorithm.

CF-SCD = Coverage Fingerprint - Spatial Change Detection, as described in the
"He thong thong minh ung dung Coverage Fingerprint va Tri tue nhan tao trong
giam sat bien dong vung phu" proposal.

This package turns UE Report + cell RF/installation data (+ Alarm) into:
  1. Coverage Fingerprints (per cell, per observation cycle)
  2. Historical baselines (mean / median / MAD)
  3. Spatial change detection (rule-based + SSI + time-series, ensemble voting)
  4. Root-cause analysis (neighbor correlation + expert rule library + alarms)
  5. Coverage Health Score (0-100, aggregated across the network hierarchy)
"""

from .config import CFSCDConfig
from .pipeline import CoverageIntelligencePipeline

__all__ = ["CFSCDConfig", "CoverageIntelligencePipeline"]
