"""Task registry of the decoupled framework (importing a task registers it)."""

from rl.decoupled.tasks import lower_body  # noqa: F401
from rl.decoupled.tasks import joint_foot_mimic  # noqa: F401
from rl.decoupled.tasks import joint_foot_mimic_arm  # noqa: F401
from rl.decoupled.tasks import joint_vel  # noqa: F401
from rl.decoupled.tasks import joint_vel_mpc  # noqa: F401
from rl.decoupled.tasks import joint_foot_mpc  # noqa: F401
from rl.decoupled.tasks import joint_vel_force_mpc  # noqa: F401
from rl.decoupled.tasks import joint_vel_vforce_mpc  # noqa: F401
from rl.decoupled.tasks import joint_foot_hlip  # noqa: F401
from rl.decoupled.tasks import joint_vel_hlip  # noqa: F401
from rl.decoupled.tasks import joint_vel_hlip_mpc  # noqa: F401
