"""Classes and functions for collecting and storing run statistics """
import abc
import time
import itertools
from collections import ChainMap
from typing import Mapping, Any, List, Sequence, Iterable, Dict, Optional

import numpy as np

from intmcp import tree
import intmcp.model as M
from intmcp import policy

# pylint: disable=[no-self-use]

AgentStatisticsMap = Mapping[M.AgentID, Mapping[str, Any]]
NestedStatistic = Dict[tree.NestingLevel, Dict[M.AgentID, float]]
NestedStatisticList = Dict[tree.NestingLevel, Dict[M.AgentID, List[float]]]

# The policy classes that implement a nested reasoning structure
NESTED_POLICIES = (tree.NestedSearchTree,)


def generate_episode_statistics(trackers: Iterable['Tracker']
                                ) -> AgentStatisticsMap:
    """Generate episode statistics from set of trackers """
    statistics = combine_statistics([t.get_episode() for t in trackers])
    return statistics


def generate_statistics(trackers: Iterable['Tracker']) -> AgentStatisticsMap:
    """Generate summary statistics from set of trackers """
    statistics = combine_statistics([t.get() for t in trackers])
    return statistics


def combine_statistics(statistic_maps: Sequence[AgentStatisticsMap]
                       ) -> AgentStatisticsMap:
    """Combine multiple Agent statistic maps into a single one """
    agent_ids = list(statistic_maps[0].keys())
    return {
        i: dict(ChainMap(*(stat_maps[i] for stat_maps in statistic_maps)))
        for i in agent_ids
    }


def calculate_entropy(dist: np.ndarray) -> float:
    """Calculate entropy of a distribution """
    return -np.sum(np.where(
        dist != 0, dist * np.log2(dist), 0
    ))


def calculate_kl_divergence(dist1: np.ndarray, dist2: np.ndarray) -> float:
    """Calculate the KL divergence between two distributions """
    return np.sum(np.where(
        dist1 != 0, dist1 * np.log(dist1 / dist2), 0
    ))


def get_default_trackers(gamma: float,
                         policies: Sequence[policy.BasePolicy]
                         ) -> Sequence['Tracker']:
    """Get the default set of Trackers """
    num_agents = len(policies)
    trackers = [
        EpisodeTracker(num_agents, [gamma] * num_agents),
        SearchTimeTracker(num_agents),
        PolicyEntropyTracker(num_agents)
    ]

    return trackers


class Tracker(abc.ABC):
    """Generic Tracker Base class """

    @abc.abstractmethod
    def step(self,
             env: M.POSGModel,
             timestep: M.JointTimestep,
             action: M.JointAction,
             policies: Sequence[policy.BasePolicy],
             episode_end: bool) -> None:
        """Accumulates statistics for a single step """

    @abc.abstractmethod
    def reset(self) -> None:
        """Resets all gathered statistics """

    @abc.abstractmethod
    def reset_episode(self) -> None:
        """Resets all statistics prior to each episode """

    @abc.abstractmethod
    def get_episode(self) -> AgentStatisticsMap:
        """Aggregates episode statistics for each agent """

    @abc.abstractmethod
    def get(self) -> AgentStatisticsMap:
        """Aggregates all episode statistics for each agent """


class EpisodeTracker(Tracker):
    """Tracks episode return and other statistics """

    def __init__(self, num_agents: int, discounts: List[float]):
        assert len(discounts) == num_agents
        self._num_agents = num_agents
        self._discounts = np.array(discounts)

        self._num_episodes = 0
        self._current_episode_done = False
        self._current_episode_start_time = time.time()
        self._current_episode_returns = np.zeros(num_agents)
        self._current_episode_discounted_returns = np.zeros(num_agents)
        self._current_episode_steps = 0

        self._dones = []                # type: ignore
        self._times = []                # type: ignore
        self._returns = []              # type: ignore
        self._discounted_returns = []   # type: ignore
        self._steps = []                # type: ignore
        self._outcomes = []             # type: ignore

    def step(self,
             env: M.POSGModel,
             timestep: M.JointTimestep,
             action: M.JointAction,
             policies: Sequence[policy.BasePolicy],
             episode_end: bool) -> None:
        next_state, _, rewards, done = timestep
        self._current_episode_returns += rewards
        self._current_episode_discounted_returns += (
            self._discounts**self._current_episode_steps * rewards
        )
        self._current_episode_done = done
        self._current_episode_steps += 1

        if episode_end:
            self._num_episodes += 1
            self._dones.append(self._current_episode_done)
            self._times.append(time.time() - self._current_episode_start_time)
            self._returns.append(self._current_episode_returns)
            self._discounted_returns.append(
                self._current_episode_discounted_returns
            )
            self._steps.append(self._current_episode_steps)
            self._outcomes.append(env.get_outcome(next_state))

    def reset(self) -> None:
        self.reset_episode()
        self._num_episodes = 0
        self._dones = []
        self._times = []
        self._returns = []
        self._discounted_returns = []
        self._steps = []
        self._outcomes = []

    def reset_episode(self) -> None:
        self._current_episode_done = False
        self._current_episode_start_time = time.time()
        self._current_episode_returns = np.zeros(self._num_agents)
        self._current_episode_discounted_returns = np.zeros(self._num_agents)
        self._current_episode_steps = 0

    def get_episode(self) -> AgentStatisticsMap:
        stats = {}
        for i in range(self._num_agents):
            stats[i] = {
                "episode_number": self._num_episodes,
                "episode_return": self._returns[-1][i],
                "episode_discounted_return": (
                    self._discounted_returns[-1][i]
                ),
                "episode_steps": self._steps[-1],
                "episode_outcome": self._outcomes[-1][i],
                "episode_done": self._dones[-1],
                "episode_time": self._times[-1]
            }

        return stats

    def get(self) -> AgentStatisticsMap:
        outcome_counts = {
            k: [0 for _ in range(self._num_agents)] for k in M.Outcomes
        }
        for outcome in self._outcomes:
            for i in range(self._num_agents):
                outcome_counts[outcome[i]][i] += 1

        stats = {}
        for i in range(self._num_agents):
            stats[i] = {
                "num_episodes": self._num_episodes,
                "episode_returns_mean": np.mean(self._returns, axis=0)[i],
                "episode_returns_std": np.std(self._returns, axis=0)[i],
                "episode_returns_max": np.max(self._returns, axis=0)[i],
                "episode_returns_min": np.min(self._returns, axis=0)[i],
                "episode_discounted_returns_mean": (
                    np.mean(self._discounted_returns, axis=0)[i]
                ),
                "episode_discounted_returns_std": (
                    np.std(self._discounted_returns, axis=0)[i]
                ),
                "episode_discounted_returns_max": (
                    np.max(self._discounted_returns, axis=0)[i]
                ),
                "episode_discounted_returns_min": (
                    np.min(self._discounted_returns, axis=0)[i]
                ),
                "episode_steps_mean": np.mean(self._steps),
                "episode_steps_std": np.std(self._steps),
                "episode_times_mean": np.mean(self._times),
                "episode_times_std": np.std(self._times),
                "episode_dones": np.mean(self._dones)
            }

            for outcome, counts in outcome_counts.items():
                stats[i][f"num_outcome_{outcome}"] = counts[i]

        return stats


class SearchTimeTracker(Tracker):
    """Tracks Search, Update, Reinvigoration time in Search Trees """

    # The list of keys to track from the policies.statistics property
    TIME_KEYS = [
        "search_time",
        "update_time",
        "reinvigoration_time"
    ]

    def __init__(self, num_agents: int):
        self._num_agents = num_agents

        self._num_episodes = 0
        self._current_episode_steps = 0
        self._current_episode_times: Dict[
            M.AgentID, Dict[str, List[float]]
        ] = {}

        self._steps: List[int] = []
        self._times: Dict[M.AgentID, Dict[str, List[float]]] = {}

        self.reset()

    def step(self,
             env: M.POSGModel,
             timestep: M.JointTimestep,
             action: M.JointAction,
             policies: Sequence[policy.BasePolicy],
             episode_end: bool) -> None:
        self._current_episode_steps += 1

        for i in range(self._num_agents):
            statistics = policies[i].statistics
            for time_key in self.TIME_KEYS:
                self._current_episode_times[i][time_key].append(
                    statistics.get(time_key, 0.0)
                )

        if episode_end:
            self._num_episodes += 1
            self._steps.append(self._current_episode_steps)
            for i in range(self._num_agents):
                for k in self.TIME_KEYS:
                    key_step_times = self._current_episode_times[i][k]
                    self._times[i][k].append(np.mean(key_step_times))

    def reset(self) -> None:
        self.reset_episode()
        self._num_episodes = 0
        self._steps = []
        self._times = {}
        for i in range(self._num_agents):
            self._times[i] = {k: [] for k in self.TIME_KEYS}

    def reset_episode(self) -> None:
        self._current_episode_steps = 0
        self._current_episode_times = {}
        for i in range(self._num_agents):
            self._current_episode_times[i] = {k: [] for k in self.TIME_KEYS}

    def get_episode(self) -> AgentStatisticsMap:
        stats = {}
        for i in range(self._num_agents):
            agent_stats = {}
            for key, step_times in self._current_episode_times[i].items():
                agent_stats[key] = np.mean(step_times, axis=0)
            stats[i] = agent_stats

        return stats

    def get(self) -> AgentStatisticsMap:
        stats = {}
        for i in range(self._num_agents):
            agent_stats = {}
            for key, values in self._times[i].items():
                agent_stats[f"{key}_mean"] = np.mean(values, axis=0)
                agent_stats[f"{key}_std"] = np.std(values, axis=0)
            stats[i] = agent_stats
        return stats


class PolicyEntropyTracker(Tracker):
    """Tracks the entropy of each agent's policy """

    def __init__(self, num_agents: int):
        self._num_agents = num_agents

        self._num_episodes = 0
        self._current_episode_steps = 0
        self._current_episode_entropies: Dict[M.AgentID, List[float]] = {}

        self._steps: List[int] = []
        self._entropies: Dict[M.AgentID, List[float]] = {}

        self.reset()

    def step(self,
             env: M.POSGModel,
             timestep: M.JointTimestep,
             action: M.JointAction,
             policies: Sequence[policy.BasePolicy],
             episode_end: bool) -> None:
        self._current_episode_steps += 1

        for i in range(self._num_agents):
            action_dist = policies[i].get_pi_by_history()
            entropy = self._calculate_entropy(action_dist)
            self._current_episode_entropies[i].append(entropy)

        if episode_end:
            self._num_episodes += 1
            self._steps.append(self._current_episode_steps)
            for i in range(self._num_agents):
                entropies = self._current_episode_entropies[i]
                if len(entropies) == 0:
                    mean_entropy = 0.0
                else:
                    mean_entropy = np.mean(entropies)
                self._entropies[i].append(mean_entropy)

    def _calculate_entropy(self, action_dist: policy.ActionDist) -> float:
        dist = np.array(list(action_dist.values()), dtype=np.float64)
        return calculate_entropy(dist)

    def reset(self) -> None:
        self.reset_episode()
        self._num_episodes = 0
        self._steps = []
        self._entropies = {i: [] for i in range(self._num_agents)}

    def reset_episode(self) -> None:
        self._current_episode_steps = 0
        self._current_episode_entropies = {
            i: [] for i in range(self._num_agents)
        }

    def get_episode(self) -> AgentStatisticsMap:
        stats = {}
        for i in range(self._num_agents):
            step_entropies = self._current_episode_entropies[i]
            stats[i] = {
                "policy_entropy_mean": np.mean(step_entropies),
                "policy_entropy_std": np.std(step_entropies)
            }
        return stats

    def get(self) -> AgentStatisticsMap:
        stats = {}
        for i in range(self._num_agents):
            episode_entropies = self._entropies[i]
            stats[i] = {
                "policy_entropy_mean": np.mean(episode_entropies),
                "policy_entropy_std": np.std(episode_entropies)
            }
        return stats


class PolicyKLTracker(Tracker):
    """Tracks KL-Divergence between agents predicted policy and the true policy

    Specifically, tracks:
    - KL divergence between the actual opponent policy and the predicted
      opponents policy
    """

    def __init__(self, num_agents: int):
        self._num_agents = num_agents

        self._num_episodes = 0
        self._current_episode_steps = 0
        self._current_episode_kls: Dict[
            M.AgentID, Dict[M.AgentID, List[float]]
        ] = {}

        self._steps = []               # type: ignore
        self._kls = {}                 # type: ignore

        self.reset()

    def step(self,
             env: M.POSGModel,
             timestep: M.JointTimestep,
             action: M.JointAction,
             policies: Sequence[policy.BasePolicy],
             episode_end: bool) -> None:
        self._current_episode_steps += 1

        for i in range(self._num_agents):
            kl_divergences = self._get_kl_divergences(i, policies)
            for j, kl in kl_divergences.items():
                self._current_episode_kls[i][j].append(kl)

        if episode_end:
            self._num_episodes += 1
            self._steps.append(self._current_episode_steps)

            for i in range(self._num_agents):
                step_kls = self._current_episode_kls[i]
                for j, kls in step_kls.items():
                    if len(kls) == 0:
                        mean_kls = 0.0
                    else:
                        mean_kls = np.mean(kls)
                    self._kls[i][j].append(mean_kls)

    def _get_kl_divergences(self,
                            agent_id: M.AgentID,
                            policies: Sequence[policy.BasePolicy]
                            ) -> Dict[M.AgentID, float]:
        agent_policy = policies[agent_id]
        kl_divergences = {}
        if not isinstance(agent_policy, NESTED_POLICIES):
            for j in range(self._num_agents):
                if j == agent_id:
                    continue
                kl_divergences[j] = float('inf')
            return kl_divergences

        if isinstance(agent_policy, tree.NestedSearchTree):
            nested_pis = agent_policy.get_nested_pis(nesting_depth_limit=1)
            other_nesting_level = agent_policy.nesting_level - 1

            if other_nesting_level not in nested_pis:
                return kl_divergences

            other_pis = nested_pis[other_nesting_level]
            for j in range(self._num_agents):
                if j == agent_id:
                    continue
                actual_policy = policies[j].get_pi_by_history()
                predicted_policy = other_pis[j]
                kl_divergences[j] = self._calculate_kl_divergence(
                    actual_policy, predicted_policy
                )

        return kl_divergences

    def _calculate_kl_divergence(self,
                                 true_action_dist: policy.ActionDist,
                                 pred_action_dist: policy.ActionDist) -> float:
        true_dist = np.array(list(true_action_dist.values()), dtype=np.float64)
        pred_dist = np.array(list(pred_action_dist.values()), dtype=np.float64)
        return calculate_kl_divergence(true_dist, pred_dist)

    def reset(self) -> None:
        self.reset_episode()
        self._num_episodes = 0
        self._steps = []
        self._kls = {}
        for i in range(self._num_agents):
            self._kls[i] = {j: [] for j in range(self._num_agents)}

    def reset_episode(self) -> None:
        self._current_episode_steps = 0
        self._current_episode_kls = {}
        for i in range(self._num_agents):
            self._current_episode_kls[i] = {
                j: [] for j in range(self._num_agents)
            }

    def get_episode(self) -> AgentStatisticsMap:
        stats = {}
        for i in range(self._num_agents):
            agent_stats = {}

            step_kls = self._current_episode_kls[i]
            for j in range(self._num_agents):
                header = f"policy_kl_{j}"
                if j == i:
                    kl_mean = 0.0
                    kl_std = 0.0
                else:
                    kls = step_kls[j]
                    kl_mean = np.mean(kls)
                    if kl_mean == float('inf'):
                        kl_std = 0.0
                    else:
                        kl_std = np.std(kls)
                agent_stats[f"{header}_mean"] = kl_mean
                agent_stats[f"{header}_std"] = kl_std

            stats[i] = agent_stats

        return stats

    def get(self) -> AgentStatisticsMap:
        """Aggregates all episode statistics for each agent """
        stats = {}
        for i in range(self._num_agents):
            agent_stats = {}

            episode_kls = self._kls[i]
            for j in range(self._num_agents):
                header = f"policy_kl_{j}"
                if j == i:
                    kl_mean = 0.0
                    kl_std = 0.0
                else:
                    kls = episode_kls[j]
                    kl_mean = np.mean(kls)
                    if kl_mean == float('inf'):
                        kl_std = 0.0
                    else:
                        kl_std = np.std(kls)
                agent_stats[f"{header}_mean"] = kl_mean
                agent_stats[f"{header}_{j}_std"] = kl_std

            stats[i] = agent_stats
        return stats


class NestedBeliefEntropyTracker(Tracker):
    """Tracks entropy of agent's nested beliefs at each level of nesting

    Only tracks statistics for Nested Policies (i.e. NestedSearchTree)
    """

    def __init__(self,
                 num_agents: int,
                 policies: Sequence[policy.BasePolicy]):
        self._num_agents = num_agents

        self._nesting_levels: List[Optional[tree.NestingLevel]] = []
        for pi in policies:
            if isinstance(pi, tree.NestedSearchTree):
                self._nesting_levels.append(pi.nesting_level)
            else:
                self._nesting_levels.append(None)

        self._max_nesting_level: Optional[int] = None
        if any(nl is not None for nl in self._nesting_levels):
            self._max_nesting_level = max(
                [nl for nl in self._nesting_levels if nl is not None]
            )

        self._num_episodes = 0
        self._current_episode_steps = 0
        self._current_episode_entropies = {}    # type: ignore

        self._steps = []               # type: ignore
        self._entropies = {}           # type: ignore

        self.reset()

    def step(self,
             env: M.POSGModel,
             timestep: M.JointTimestep,
             action: M.JointAction,
             policies: Sequence[policy.BasePolicy],
             episode_end: bool) -> None:
        self._current_episode_steps += 1

        for i in range(self._num_agents):
            policy_i = policies[i]
            if isinstance(policy_i, tree.NestedSearchTree):
                nested_belief = policy_i.get_nested_state_beliefs()
                self._step_nested_entropies(i, nested_belief)

        if episode_end:
            self._num_episodes += 1
            self._steps.append(self._current_episode_steps)
            for i in range(self._num_agents):
                self._end_episode_entropies(i)

    def _step_nested_entropies(self,
                               i: M.AgentID,
                               nested_belief: tree.NestedBelief) -> None:
        nested_entropies = self._calc_nested_entropies(nested_belief)
        for nl, agent_entropies in nested_entropies.items():
            for j, entropy in agent_entropies.items():
                self._current_episode_entropies[i][nl][j].append(entropy)

    def _calc_nested_entropies(self,
                               nested_belief: tree.NestedBelief,
                               ) -> NestedStatistic:
        nested_entropies = {}
        for nl, agent_beliefs in nested_belief.items():
            agent_entropies = {}
            for i, belief in agent_beliefs.items():
                if len(belief) == 0:
                    agent_entropies[i] = 0.0
                else:
                    dist = np.array(list(belief.values()), dtype=np.float64)
                    agent_entropies[i] = calculate_entropy(dist)
            nested_entropies[nl] = agent_entropies
        return nested_entropies

    def _end_episode_entropies(self, i: M.AgentID) -> None:
        nested_step_entropies = self._current_episode_entropies[i]
        for nl, agent_step_entropies in nested_step_entropies.items():
            for j, step_entropies in agent_step_entropies.items():
                if len(step_entropies) == 0:
                    mean_entropy = 0.0
                else:
                    mean_entropy = np.mean(step_entropies)
                self._entropies[i][nl][j].append(mean_entropy)

    def reset(self) -> None:
        self.reset_episode()
        self._num_episodes = 0
        self._steps = []
        self._entropies = {}
        for i, nl_i in enumerate(self._nesting_levels):
            self._entropies[i] = self._init_nested_statistic(nl_i)

    def reset_episode(self) -> None:
        self._current_episode_steps = 0
        self._current_episode_entropies = {}
        for i, nl_i in enumerate(self._nesting_levels):
            nested_entropies = self._init_nested_statistic(nl_i)
            self._current_episode_entropies[i] = nested_entropies

    def _init_nested_statistic(self,
                               nesting_level: Optional[int]
                               ) -> NestedStatisticList:
        nested_statistic_list = {}    # type: ignore
        if nesting_level is None:
            return nested_statistic_list

        for nl in range(nesting_level + 1):
            nested_statistic_list[nl] = {
                j: [] for j in range(self._num_agents)
            }
        return nested_statistic_list

    def get_episode(self) -> AgentStatisticsMap:
        stats = {}

        for i in range(self._num_agents):
            agent_stats = {}
            entropy_statistics = self._get_nested_statistic(
                "belief_entropy", self._current_episode_entropies[i]
            )
            agent_stats.update(entropy_statistics)

            stats[i] = agent_stats

        return stats

    def get(self) -> AgentStatisticsMap:
        """Aggregates all episode statistics for each agent """
        stats = {}
        for i in range(self._num_agents):
            agent_stats = {}
            entropy_statistics = self._get_nested_statistic(
                "belief_entropy", self._entropies[i]
            )
            agent_stats.update(entropy_statistics)

            stats[i] = agent_stats
        return stats

    def _get_nested_statistic(self,
                              prefix: str,
                              nested_statistics: NestedStatisticList
                              ) -> Dict[str, float]:
        if self._max_nesting_level is None:
            return {}

        flat_statistics = {}      # type: ignore
        for nl, j in itertools.product(
                range(self._max_nesting_level+1),
                range(self._num_agents)
        ):
            header = f"{prefix}_{nl}_{j}"
            if nl not in nested_statistics or j not in nested_statistics[nl]:
                statistic_mean = np.NAN
                statistic_std = np.NAN
            elif len(nested_statistics[nl][j]) == 0:
                statistic_mean = 0.0
                statistic_std = 0.0
            else:
                step_statistics = nested_statistics[nl][j]
                statistic_mean = np.mean(step_statistics)
                statistic_std = np.std(step_statistics)
            flat_statistics[f"{header}_mean"] = statistic_mean
            flat_statistics[f"{header}_std"] = statistic_std

        return flat_statistics
