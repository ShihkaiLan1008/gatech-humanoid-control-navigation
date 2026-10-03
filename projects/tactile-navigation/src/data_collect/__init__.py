"""Room-2 touch-transition collection: a dataset, not a demo.

This package records `(x_t, M, c_t, o_{t+1}, x_{t+1})` transitions for training
a tactile world model.  It shares the room, the controller and the safety
stack with the navigation demo and shares NOTHING with its decision making:
no planner, no expected-information-gain, no policy picks a base pose or a
touch target here.  Poses come from a frozen route centreline plus a uniform
lateral jitter; targets come from a stratified uniform draw over the
controller's own trained workspace.

The one place the simulator's own geometry is consulted is the sampler's
`predict_contact`, which only decides WHICH randomly drawn candidate to spend
a touch on so the finished dataset is not 90 % misses.  The label always comes
from executing the controller.
"""
