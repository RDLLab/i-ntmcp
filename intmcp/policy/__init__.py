from .base_policy import BasePolicy, ActionDist, StateDist
from .random_policy import RandomPolicy


# Set of available general policy implementations that work with any env
POLICIES = {
    "random": RandomPolicy
}
