"""DAIRLab-style active tactile exploration, ported for landmark localization.

PROVENANCE.  ``noise.py``, ``belief.py`` and ``infogain.py`` are verbatim
ports from branch ``wt/9.4-tp`` (planner/tactile_explore, commit 9a17a4f) --
the validated ATE program (privilege audit 12/12; belief and EIG unit tests on
that branch).  Nothing here re-derives them; changes, if ever needed, go
through that program first.

WHAT THIS COPY IS FOR.  On ``hw-robustness`` the ATE acts as the LOCAL PLANNER
at landmarks: the executor instantiates a one-hypothesis Belief for the
landmark's known shape (wall corner, cylinder), and the EIG scorer decides
where to press, how many times, and when to stop.  The navigation particle
filter stays the pose belief; ATE selects, the PF localizes, both consume the
same touches.
"""
