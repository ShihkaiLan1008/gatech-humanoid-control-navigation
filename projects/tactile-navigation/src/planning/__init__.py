"""Geometric route generation.

Answers one question -- what collision-free routes exist between two points on a
known map -- and answers nothing about where to touch.  The belief-aware planner
consumes the routes; it is not consumed by them.
"""

from .geometric import (OccupancyGrid, Route, RouteConfig, plan_alternatives,
                        plan_route)

__all__ = ["OccupancyGrid", "Route", "RouteConfig", "plan_route", "plan_alternatives"]
