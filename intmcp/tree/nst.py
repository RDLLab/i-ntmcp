"""The base NestedSearchTree tree class """
import time
import math
import random
from argparse import ArgumentParser
from typing import Optional, List, Dict, Collection, Union, Tuple, Type

import intmcp.model as M
from intmcp import policy

from intmcp.tree.node import Node
from intmcp.tree.history_state import HistoryState

# pylint: disable=[protected-access]

NestingLevel = int
NestedBelief = Dict[NestingLevel, Dict[M.AgentID, policy.StateDist]]
HistoryDist = Dict[M.AgentHistory, float]
RolloutPolicyMap = Dict[M.AgentID, Tuple[Type[policy.BasePolicy], Dict]]


class NestedSearchTree(policy.BasePolicy):
    """Nested Search Tree using UCT Action selection

    In this tree, nested trees are constructed sequentially, bottom up. That is
    the 0th level tree is expanded first, then the 1st level tree, and so.

    The number of simulations assigned to each level is specified by the nested
    search schedule. By default it is uniform, so each nested tree recieves
    the same amount of resources for expansion.
    """

    # Limits on rejection sampling for belief reinvigoration
    SAMPLE_RETRY_LIMIT = 1000
    SAMPLE_FAIL_LIMIT = 3

    # Initial value for node visit count for preferred actions
    PREFERRED_N_INIT = 10

    def __init__(self,
                 model: M.POSGModel,
                 ego_agent: M.AgentID,
                 nesting_level: NestingLevel,
                 nested_trees: Dict[M.AgentID, 'NestedSearchTree'],
                 num_sims: int,
                 rollout_policy: policy.BasePolicy,
                 uct_c: float,
                 gamma: float,
                 reinvigorate_fn: Optional[M.ReinvigorateFunction],
                 extra_particles_prop: float = 1.0 / 16,
                 step_limit: Optional[int] = None,
                 epsilon: float = 0.01,
                 **kwargs):
        super().__init__(model, ego_agent, gamma, **kwargs)
        assert nesting_level == 0 or len(nested_trees) == model.num_agents-1
        assert extra_particles_prop > 0

        self.num_agents = model.num_agents
        self.nesting_level = nesting_level
        self.nested_trees = nested_trees
        self.num_sims = num_sims
        self._rollout_policy = rollout_policy
        self._uct_c = uct_c
        self._reinvigorate_fn = reinvigorate_fn
        self._extra_particles = math.ceil(num_sims * extra_particles_prop)
        self._step_limit = step_limit
        self._epsilon = epsilon
        if gamma == 0.0:
            self._depth_limit = 0
        else:
            self._depth_limit = math.ceil(math.log(epsilon) / math.log(gamma))

        # Initialize initial belief
        b_0 = HistoryState.initial_belief(
            self.model.initial_belief, self.num_agents
        )
        self._initial_belief = b_0
        self.root = Node(None, None, b_0)

        self._step_num = 0
        self._statistics = {}
        self._reset_step_statistics()

    #######################################################
    # Step
    #######################################################

    def step(self, obs: M.Observation) -> M.Action:
        assert self._step_limit is None or self._step_num <= self._step_limit
        self._reset_step_statistics()
        self._log_info1(f"Step obs={obs}")
        if self._step_num == 0:
            self._init_update(obs)
        else:
            self.update(self._last_action, obs)

        self._log_info1(f"history={self.history}")

        self._last_action = self.get_action()
        self._step_num += 1
        self._statistics.update(self._collect_nested_statistics())
        return self._last_action

    def _collect_nested_statistics(self) -> Dict:
        # This functions adds up statistics of each NST in the hierarchy
        # Only adds statistics that are collected at each level, so doesn't
        # add up 'search_time' or 'update_time' which are only collected
        # in the top tree
        stats = {
            "reinvigoration_time": self._statistics["reinvigoration_time"]
        }
        if self.nesting_level > 0:
            for pi in self.nested_trees.values():
                nested_stats = pi._collect_nested_statistics()
                for key in stats:
                    stats[key] += nested_stats[key]
        return stats

    #######################################################
    # RESET
    #######################################################

    def reset(self) -> None:
        self._log_info1("Reset")
        self._step_num = 0
        self.history = M.AgentHistory.get_init_history()
        self.root = Node(None, None, self._initial_belief)
        self._last_action = M.Action.get_null_action()
        for nested_tree in self.nested_trees.values():
            nested_tree.reset()
        self._reset_step_statistics()

    def _reset_step_statistics(self):
        self._statistics = {
            "search_time": 0.0,
            "update_time": 0.0,
            "reinvigoration_time": 0.0
        }

    #######################################################
    # UPDATE
    #######################################################

    def _init_update(self, obs: M.Observation):
        self._log_info1("Initial update")
        start_time = time.time()

        self.history = M.AgentHistory.get_init_history(obs)
        self._last_action = M.Action.get_null_action()
        self._initial_nested_update({self.history: 1.0})

        update_time = time.time() - start_time
        self._statistics["update_time"] = update_time

    def _initial_nested_update(self, histories: Dict[M.AgentHistory, float]):
        """Nested reset """
        self._log_info2(f"Nested Reset for num_histories={len(histories)}")

        for hist, h_prob in histories.items():
            h_node = self.traverse(hist)
            _, init_obs = hist.get_last_step()
            b_0 = self.model.get_init_belief(self.ego_agent, init_obs)

            # total particles in initial beliefs should match num sims
            num_samples = math.ceil(h_prob*self.num_sims)
            hs_b_0 = M.ParticleBelief()
            for _ in range(num_samples):
                state = b_0.sample()
                joint_obs = self.model.sample_initial_obs(state)
                joint_history = M.JointHistory.get_init_history(
                    self.num_agents, joint_obs
                )
                hs_b_0.add_particle(HistoryState(state, joint_history))

            h_node.belief = hs_b_0

        if self.nesting_level > 0:
            nested_histories = self.get_nested_history_dist(histories)
            self._log_debug("Recursing nested initial update to next level")
            self._log_debug(str(nested_histories))
            for i, nested_tree in self.nested_trees.items():
                nested_tree._initial_nested_update(nested_histories[i])

    def update(self, action: M.Action, obs: M.Observation) -> None:
        self._log_info1(f"Updating with a={action} o={obs}")
        start_time = time.time()

        self.history = self.history.extend(action, obs)
        horizon = len(self.history.history)
        self._nested_update({self.history: 1.0}, horizon)

        update_time = time.time() - start_time
        self._statistics["update_time"] = update_time
        self._log_info1(f"Update time = {update_time:.4f}s")

    def _nested_update(self,
                       histories: Dict[M.AgentHistory, float],
                       horizon: int) -> None:
        """Update nested tree using joint history distribution """
        self._log_info2(f"Nested Update num_histories={len(histories)}")
        self._prune(histories, horizon)

        self._reinvigorate_histories(histories)

        if self.nesting_level > 0:
            nested_histories = self.get_nested_history_dist(histories)
            self._log_debug("Recursing nested update down to next level")
            for i, nested_tree in self.nested_trees.items():
                nested_tree._nested_update(nested_histories[i], horizon)

    def _prune(self, histories: Collection[M.AgentHistory], horizon: int):
        """Prune histories from tree that are not in collection """
        self._log_debug("Pruning histories")

        for action_node in self.root.children:
            for obs_node in action_node.children:
                h = M.AgentHistory(((action_node.h, obs_node.h),))
                self._prune_traverse(obs_node, h, histories, horizon)

    def _prune_traverse(self,
                        node: Node,
                        node_history: M.AgentHistory,
                        histories: Collection[M.AgentHistory],
                        horizon: int):
        """Recursively Traverse and prune histories from tree """
        if len(node_history.history) == horizon:
            if node_history not in histories:
                self._log_debug1(f"{horizon=} pruning history={node_history}")
                del node
            return

        for action_node in node.children:
            for obs_node in action_node.children:
                history_tp1 = node_history.extend(action_node.h, obs_node.h)

                if horizon-2 > 0 and len(history_tp1.history) == horizon-2:
                    self._log_debug2(
                        f"{horizon=} clearing obs node for "
                        f"t={len(history_tp1.history)}"
                    )
                    # clear particles from unused nodes to reduce mem usage
                    obs_node.clear_belief()

                self._prune_traverse(obs_node, history_tp1, histories, horizon)

    def get_nested_history_dist(self,
                                histories: Dict[M.AgentHistory, float],
                                ) -> Dict[int, HistoryDist]:
        """Get dist over nested hists based on higher level history dist """
        self._log_debug1(f"get_nested_history_dist {histories}")
        nested_histories: Dict = {}
        for hist, h_prob in histories.items():
            h_node = self.traverse(hist)
            for i in self.nested_trees:
                if i not in nested_histories:
                    nested_histories[i] = {}
                nested_histories[i] = self.update_history_dist(
                    h_node, h_prob, nested_histories[i], i
                )
        return nested_histories

    def update_history_dist(self,
                            h_node: Node,
                            h_prob: float,
                            histories: Dict[M.AgentHistory, float],
                            agent_id: M.AgentID,
                            ) -> HistoryDist:
        """Update history distribution of given Node, or create new one if
        existing distribution is not provided
        """
        if h_node.is_root():
            belief_size = math.ceil(
                h_prob * (self.num_sims+self._extra_particles)
            )
            particles = h_node.belief.sample_k(belief_size)
        elif h_node.belief.size() == 0:
            return histories
        else:
            assert isinstance(h_node.belief, M.ParticleBelief)
            belief_size = h_node.belief.size()
            particles = h_node.belief.particles

        h_count = {}
        for h_state in particles:
            assert isinstance(h_state, HistoryState)
            h = h_state.history.get_agent_history(agent_id)

            if h not in h_count:
                h_count[h] = 1.0
            else:
                h_count[h] += 1.0

        for hist, count in h_count.items():
            if hist not in histories:
                histories[hist] = 0.0
            histories[hist] += h_prob * (count / belief_size)

        return histories

    #######################################################
    # SEARCH
    #######################################################

    def get_action(self) -> M.Action:
        """Get action given current tree root belief """
        self._log_info1(f"Searching for num_sims={self.num_sims}")
        start_time = time.time()

        for level in range(self.nesting_level+1):
            num_level_sims = self._get_level_sims(level)
            self._log_info2(f"Searching {level=} for {num_level_sims=}")
            for _ in range(num_level_sims):
                self._nested_sim(self.history, level, True)

        search_time = time.time() - start_time
        search_time_per_sim = search_time / self.num_sims
        self._statistics["search_time"] += search_time
        self._log_info1(f"{search_time=:.2f} s, {search_time_per_sim=:.5f}")

        return self.get_action_by_history(self.history)

    def get_action_init_values(self,
                               history: M.AgentHistory
                               ) -> Dict[M.Action, Tuple[float, int]]:
        # abstract method inherited from BasePolicy parent class
        # Not used for this class
        raise NotImplementedError

    def get_value(self, history: Optional[M.AgentHistory]) -> float:
        # abstract method inherited from BasePolicy parent class
        # Not used for this class
        raise NotImplementedError

    def _get_level_sims(self, nesting_level: NestingLevel) -> int:
        """Get the number of sims to run at a given level.

        Assumes that all nested trees in a level share the same number of sims.
        """
        if nesting_level == self.nesting_level:
            return self.num_sims
        nested_agent = list(self.nested_trees)[0]
        return self.nested_trees[nested_agent]._get_level_sims(nesting_level)

    def _nested_sim(self,
                    history: M.AgentHistory,
                    search_level: int,
                    top_level: bool = False):
        """Run nested simulation on this policy tree starting from history """
        self._log_debug2(f"nested_sim: {search_level=}")

        root = self.traverse(history)
        if len(root.children) == 0:
            self.expand(root, history)

        self._handle_depleted_node(history, root, top_level)

        h_state = root.belief.sample()
        assert isinstance(h_state, HistoryState)
        if self.nesting_level > search_level:
            for i, nested_tree in self.nested_trees.items():
                nested_tree._nested_sim(
                    h_state.history.get_agent_history(i), search_level, False
                )
        else:
            for i in range(self.num_agents):
                h_i = h_state.history.get_agent_history(i)
                if i == self.ego_agent:
                    self._rollout_policy.reset_history(h_i)
                elif self.nesting_level > 0:
                    self.nested_trees[i]._rollout_policy.reset_history(h_i)
            self.simulate(h_state, root, 0)
            root.n += 1

    def simulate(self,
                 h_state: HistoryState,
                 obs_node: Node,
                 depth: int) -> float:
        """Run Monte-Carlo Simulation in tree """
        if depth > self._depth_limit:
            return 0.0

        if (
            self._step_limit is not None
            and depth+self._step_num > self._step_limit
        ):
            return 0.0

        if len(obs_node.children) == 0:
            agent_history = h_state.history.get_agent_history(self.ego_agent)
            self.expand(obs_node, agent_history)
            return self.rollout(h_state.state, depth)

        joint_action = self.get_joint_sim_action(h_state, obs_node)

        next_state, joint_obs, rewards, done = self.model.step(
            h_state.state, joint_action
        )

        ego_action = joint_action.get_agent_action(self.ego_agent)
        ego_obs = joint_obs.get_agent_obs(self.ego_agent)
        ego_reward = rewards[self.ego_agent]
        new_history = h_state.history.extend(joint_action, joint_obs)
        next_h_state = HistoryState(next_state, new_history)

        action_node = obs_node.get_child(ego_action)
        child_obs_node = action_node.get_child(ego_obs)

        child_obs_node.n += 1
        child_obs_node.belief.add_particle(next_h_state)

        if not done:
            for i in range(self.num_agents):
                a_i = joint_action.get_agent_action(i)
                o_i = joint_obs.get_agent_obs(i)
                if i == self.ego_agent:
                    self._rollout_policy.update(a_i, o_i)
                elif self.nesting_level > 0:
                    self.nested_trees[i]._rollout_policy.update(a_i, o_i)

            self._rollout_policy.update(ego_action, ego_obs)
            ego_reward += self.gamma * self.simulate(
                next_h_state, child_obs_node, depth+1
            )

        action_node.n += 1
        action_node.v += (ego_reward - action_node.v) / action_node.n
        return ego_reward

    def rollout(self,
                state: M.State,
                depth: int) -> float:
        """Run Monte-Carlo Rollout """
        if depth > self._depth_limit:
            return self._rollout_policy.get_value(None)

        if (
            self._step_limit is not None
            and depth+self._step_num > self._step_limit
        ):
            return self._rollout_policy.get_value(None)

        joint_actions_list = []
        for i in range(self.num_agents):
            if i == self.ego_agent:
                a = self._rollout_policy.get_action()
            elif self.nesting_level > 0:
                a = self.nested_trees[i]._rollout_policy.get_action()
            else:
                a = random.choice(self.model.get_agent_action_space(i))
            joint_actions_list.append(a)

        joint_action = M.JointAction(tuple(joint_actions_list))
        next_state, joint_obs, rewards, done = self.model.step(
            state, joint_action
        )
        reward = rewards[self.ego_agent]
        if done:
            return reward

        for i in range(self.num_agents):
            a_i = joint_action.get_agent_action(i)
            o_i = joint_obs.get_agent_obs(i)
            if i == self.ego_agent:
                self._rollout_policy.update(a_i, o_i)
            elif self.nesting_level > 0:
                self.nested_trees[i]._rollout_policy.update(a_i, o_i)

        return reward + self.gamma * self.rollout(next_state, depth+1)

    #######################################################
    # ACTION SELECTION
    #######################################################

    def get_joint_sim_action(self,
                             h_state: HistoryState,
                             obs_node: Node) -> M.JointAction:
        """Get JointAction for simulation from given state """
        h_i = h_state.history.get_agent_history(self.ego_agent)
        ego_action = self.get_action_from_node(obs_node, h_i, False)
        return self.get_joint_action(h_state, ego_action, False)

    def get_joint_action(self,
                         h_state: HistoryState,
                         ego_action: M.Action,
                         deterministic: bool = False,
                         temperature: float = 1.0) -> M.JointAction:
        """Get JointAction from given state with given ego action. """
        agent_actions = []
        for i in range(self.num_agents):
            if i == self.ego_agent:
                action = ego_action
            elif self.nesting_level > 0:
                nested_tree = self.nested_trees[i]
                action = nested_tree.sample_action(
                    h_state.history.get_agent_history(i),
                    deterministic,
                    temperature
                )
            else:
                action = random.choice(self.model.get_agent_action_space(i))
            agent_actions.append(action)
        return M.JointAction(tuple(agent_actions))

    def get_action_by_history(self, history: M.AgentHistory) -> M.Action:
        """Get action from policy for given history """
        obs_node = self.traverse(history)
        return self.get_action_from_node(obs_node, history, True)

    def get_action_from_node(self,
                             obs_node: Node,
                             history: M.AgentHistory,
                             deterministic: bool = False) -> M.Action:
        """Get action from given node in policy tree """
        if len(obs_node.children) < len(self.action_space):
            self.expand(obs_node, history)

        if obs_node.n == 0:
            return random.choice(self.action_space)

        log_n = math.log(obs_node.n)
        max_v = -float('inf')
        max_action = obs_node.children[0].h
        for action_node in obs_node.children:
            if action_node.n == 0:
                if deterministic:
                    # want to select best explored action
                    continue
                return action_node.h
            action_v = action_node.v
            if not deterministic:
                # add exploration bonus based on relative visit count
                action_v += self._uct_c * math.sqrt(log_n / action_node.n)
            if action_v > max_v:
                max_v = action_v
                max_action = action_node.h
        return max_action

    def sample_action(self,
                      history: M.AgentHistory,
                      deterministic: bool = False,
                      temperature: float = 1.0) -> M.Action:
        """Sample an action from policy given history

        Samples using softmax function scaled based on sqrt of total
        number of visits of obs_node

        Temperature controls the spread of the softmax distribution.
        - Higher values produce more greedy sampling, while lower values
          produce more uniform sampling. Default value of 1.0 is good in
          general, but lowering the temperature below 1.0 can be useful for
          encouraging exploration (e.g. during belief reinvigoration)
        """
        obs_node = self.traverse(history)
        if deterministic:
            return self.get_action_from_node(obs_node, history, deterministic)

        if obs_node.n == 0 or len(obs_node.children) == 0:
            return random.choice(self.action_space)

        obs_n_sqrt = math.sqrt(obs_node.n)
        a_probs = [
            math.exp(a_node.n**temperature / obs_n_sqrt)
            for a_node in obs_node.children
        ]
        a_probs_sum = sum(a_probs)
        a_probs = [p / a_probs_sum for p in a_probs]

        action_node = random.choices(obs_node.children, weights=a_probs)[0]
        return action_node.h

    #######################################################
    # GENERAL METHODS
    #######################################################

    def traverse(self, history: M.AgentHistory) -> Node:
        """Traverse policy tree and return node corresponding to history """
        h_node = self.root
        for (a, o) in history:
            h_node = h_node.get_child(a).get_child(o)
        return h_node

    def expand(self, obs_node: Node, hist: M.AgentHistory):
        """Add action children to observation node in tree """
        action_init_vals = self._rollout_policy.get_action_init_values(hist)
        for action in self.action_space:
            if obs_node.has_child(action):
                continue
            v_init, n_init = action_init_vals[action]
            action_node = Node(
                action, obs_node, M.ParticleBelief(), v_init, n_init
            )
            obs_node.children.append(action_node)

    #######################################################
    # BELIEF REINVIGORATION
    #######################################################

    def _reinvigorate_histories(self, histories: Dict[M.AgentHistory, float]):
        self._log_debug(f"Reinvig beliefs for num_histories={len(histories)}")
        start_time = time.time()
        target_total_particles = self.num_sims + self._extra_particles
        for hist, h_prob in histories.items():
            h_node = self.traverse(hist)
            if h_node.parent is None:
                break
            num_samples = math.ceil(h_prob * target_total_particles)
            b_size = h_node.belief.size()
            if b_size is not None:
                num_samples -= b_size
            self._reinvigorate(hist, num_samples, h_node)
        self._log_debug(f"Reinvigorate: time={time.time()-start_time:.6f}")

    def _handle_depleted_node(self,
                              history: M.AgentHistory,
                              h_node: Node,
                              top_level: bool) -> None:
        b_size = h_node.belief.size()
        if b_size is None:
            return
        if b_size == 0 or (b_size < self._extra_particles and top_level):
            self._log_debug(
                f"depleted {h_node=} {b_size=} {top_level=}"
                f"adding {self._extra_particles} particles"
            )
            self._reinvigorate(history, self._extra_particles, h_node)

    def _reinvigorate(self,
                      history: M.AgentHistory,
                      target_node_size: int,
                      h_node: Node):
        # This function wraps the _reinvigorate_belief function and times it
        # This is necessary since the _reinvigorate_belief function can call
        # itself recursively
        self._log_debug1("Reinvigorate")
        start_time = time.time()
        self._reinvigorate_belief(history, target_node_size, h_node)
        reinvig_time = time.time() - start_time
        self._statistics["reinvigoration_time"] += reinvig_time
        self._log_debug1(f"Reinvigorate: time={reinvig_time:.6f}")

    def _reinvigorate_belief(self,
                             history: M.AgentHistory,
                             target_node_size: int,
                             h_node: Node,
                             sample_fail_limit: Optional[int] = None,
                             recurse_up: bool = True):
        """The main belief reinvigoration function.

        The general reinvigoration process:
        1. check belief needs to be reinvigorated (e.g. it's not a root belief)
        2. Reinvigorate parent if it is empty
        3. Reinvigorate node using rejection sampling/custom function for fixed
           number of tries
        4. if desired number of particles not sampled using rejection sampling/
           custom function then sample remaining particles using sampling
           without rejection
        """
        if sample_fail_limit is None:
            sample_fail_limit = self.SAMPLE_FAIL_LIMIT

        h_node_size = h_node.belief.size()
        if h_node_size is None:
            return

        num_samples = target_node_size - h_node_size
        if num_samples < 0:
            return

        parent_obs_node = h_node.parent.parent    # type: ignore
        assert parent_obs_node is not None

        # reinvigorate parent if it's size == 0
        parent_size = parent_obs_node.belief.size()
        parent_is_root = parent_obs_node.parent is None
        if (
            recurse_up
            and not parent_is_root
            and parent_size is not None
            and parent_size == 0
        ):
            self._log_debug1("Reinvigorating parent")
            self._reinvigorate_belief(
                history[:-1], target_node_size, parent_obs_node, 1, False
            )

        self._rejection_sample(
            parent_obs_node,
            h_node,
            history,
            num_samples,
            sample_fail_limit,
            force_sample=False
        )

        h_node_size = h_node.belief.size()
        if h_node_size is not None and h_node_size < target_node_size:
            self._log_debug1((
                f"Failed to reinvigorate belief after {self.SAMPLE_FAIL_LIMIT}"
                f" attempts to sample {num_samples} samples from "
                f"{parent_obs_node=} to {h_node=} for {history=}. "
                f"parent_obs_node_belief_size={parent_obs_node.belief.size()}"
            ))
            # Force sample next belief - i.e. does not reject samples
            # based on observation not matching
            # Fill belief to target node size with forced samples
            self._rejection_sample(
                parent_obs_node,
                h_node,
                history,
                target_node_size - h_node_size,
                sample_fail_limit,
                force_sample=True
            )

    def _rejection_sample(self,
                          parent_node: Node,
                          child_node: Node,
                          history: M.AgentHistory,
                          num_samples: int,
                          sample_fail_limit: int,
                          force_sample: bool = False):
        """Run rejection sampling from parent to child node

        if force_sample => does not reject based on the agent obs matching.
        """
        self._log_debug2(f"Sampling {num_samples=} for {history=}")

        sample_count = 0
        retry_count = 0
        action, obs = history[-1]

        while (
            sample_count < num_samples
            and retry_count < sample_fail_limit * self.SAMPLE_RETRY_LIMIT
        ):
            temperature = (
                1.0 / ((retry_count // self.SAMPLE_RETRY_LIMIT == 0) + 1)
            )

            h_state: HistoryState = parent_node.belief.sample()  # type: ignore
            joint_action = self.get_joint_action(
                h_state, action, temperature=temperature
            )
            next_state, joint_obs, _, _ = self.model.step(
                h_state.state, joint_action
            )

            if (
                joint_obs.get_agent_obs(self.ego_agent) != obs
                and not force_sample
            ):
                if self._reinvigorate_fn is not None:
                    try:
                        next_state, joint_obs = self._reinvigorate_fn(
                            self.ego_agent, next_state, joint_action, obs
                        )
                    except IndexError:
                        retry_count += 1
                        continue
                else:
                    retry_count += 1
                    continue

            new_history = h_state.history.extend(joint_action, joint_obs)
            next_h_state = HistoryState(next_state, new_history)
            child_node.belief.add_particle(next_h_state)   # type: ignore
            sample_count += 1

    #######################################################
    # Utility methods for analysing tree (not used for search)
    #######################################################

    def get_nested_pis(self,
                       histories: Optional[HistoryDist] = None,
                       nesting_depth_limit: Optional[int] = None
                       ) -> Dict[
                          NestingLevel, Dict[M.AgentID, policy.ActionDist]
                       ]:
        """Get nested policies """
        assert self.num_agents == 2

        if histories is None:
            histories = {self.history: 1.0}

        if nesting_depth_limit is None:
            nesting_depth_limit = self.nesting_level + 1

        ego_pi = self._get_pi_by_history_dist(histories)
        nested_pis = {self.nesting_level: {self.ego_agent: ego_pi}}

        if nesting_depth_limit == 0:
            return nested_pis

        if self.nesting_level == 0:
            j = (self.ego_agent + 1) % 2
            action_space_j = self.model.get_agent_action_space(j)
            num_actions = len(action_space_j)
            pi_j = {a: 1.0 / num_actions for a in action_space_j}
            nested_pis[-1] = {j: pi_j}
            return nested_pis

        nested_history_dists = self.get_nested_history_dist(histories)
        for i in range(self.num_agents):
            if i == self.ego_agent:
                continue

            nested_tree = self.nested_trees[i]
            agent_histories = nested_history_dists[i]
            agent_nested_pis = nested_tree.get_nested_pis(
                agent_histories, nesting_depth_limit - 1
            )
            nested_pis.update(agent_nested_pis)

        return nested_pis

    def _get_pi_by_history_dist(self,
                                history_dist: HistoryDist
                                ) -> policy.ActionDist:
        pi = {a: 0.0 for a in self.action_space}
        for history, prob in history_dist.items():
            h_pi = self.get_pi_by_history(history)
            for a in self.action_space:
                pi[a] += prob * h_pi[a]
        return pi

    def get_pi_by_history(self,
                          history: Optional[M.AgentHistory] = None
                          ) -> policy.ActionDist:
        """Get agent's distribution over actions for a given history.

        Returns the softmax distribution over actions with temperature=1.0
        (see NestedSearchTree.sample_action() function for details). This is
        used as it incorporates uncertainty based on visit counts for a given
        history.
        """
        if history is None:
            history = self.history

        obs_node = self.traverse(history)

        if obs_node.n == 0 or len(obs_node.children) == 0:
            # uniform policy
            num_actions = len(self.action_space)
            pi = {a: 1.0 / num_actions for a in self.action_space}
            return pi

        obs_n_sqrt = math.sqrt(obs_node.n)
        temp = 1.0
        pi = {
            a_node.h: math.exp(a_node.n**temp / obs_n_sqrt)
            for a_node in obs_node.children
        }

        a_probs_sum = sum(pi.values())
        for a in self.action_space:
            if a not in pi:
                pi[a] = 0.0
            pi[a] /= a_probs_sum

        return pi

    def get_belief_by_history(self,
                              history: Optional[M.AgentHistory] = None
                              ) -> policy.StateDist:
        """Get agent's distribution over states for a given history.

        May return distribution over only states with p(s) > 0
        """
        if history is None:
            history = self.history

        return self._get_state_belief_by_history_dist({history: 1.0})

    def get_nested_state_beliefs(self,
                                 histories: Optional[HistoryDist] = None
                                 ) -> NestedBelief:
        """Get nested beliefs over states

        Output is structured as:
        - Map from nesting level to
        - Map from agent ID to
        - Map from state to prob
        """
        assert self.num_agents == 2

        if histories is None:
            histories = {self.history: 1.0}

        ego_belief = self._get_state_belief_by_history_dist(histories)

        nested_state_beliefs = {
            self.nesting_level: {self.ego_agent: ego_belief}
        }

        if self.nesting_level == 0:
            return nested_state_beliefs

        nested_history_dists = self.get_nested_history_dist(histories)
        for i in range(self.num_agents):
            if i == self.ego_agent:
                continue

            nested_tree = self.nested_trees[i]
            agent_histories = nested_history_dists[i]
            agent_nested_beliefs = nested_tree.get_nested_state_beliefs(
                agent_histories
            )

            nested_state_beliefs.update(agent_nested_beliefs)

        return nested_state_beliefs

    def _get_state_belief_by_history_dist(self,
                                          history_dist: HistoryDist
                                          ) -> policy.StateDist:
        state_belief = {}
        for history, prob in history_dist.items():
            node = self.traverse(history)
            for state_history, sh_prob in node.belief.get_dist().items():
                state = state_history.state   # type: ignore
                if state not in state_belief:
                    state_belief[state] = 0.0
                state_belief[state] += prob * sh_prob
        return state_belief

    #######################################################
    # Logging
    #######################################################

    def _format_msg(self, msg: str):
        return f"i={self.ego_agent} l={self.nesting_level} {msg}"

    def __str__(self):
        return f"{self.__class__.__name__} l={self.nesting_level}"

    #######################################################
    # Class and Static Methods
    #######################################################

    @classmethod
    def get_args_parser(cls,
                        parser: Optional[ArgumentParser] = None
                        ) -> ArgumentParser:
        parser = super().get_args_parser(parser)
        parser.add_argument(
            "--uct_c", type=float, default=None,
            help="UCT C Hyperparam (default=R_max-R_min)"
        )
        parser.add_argument(
            "--nesting_level", nargs="*", type=int, default=[1],
            help="Number of nesting levels (default=[1])"
        )
        parser.add_argument(
            "--num_sims", nargs="*", type=int, default=[1000],
            help="Number of simulations to run (default=[1000])"
        )
        parser.add_argument(
            "--epsilon", type=float, default=0.01,
            help="Discount Horizon Threshold (default=0.01)"
        )
        parser.add_argument(
            "--gamma", type=float, default=0.95,
            help="Discount (default=0.95)"
        )
        return parser

    # pylint: disable=[arguments-differ]
    @classmethod
    def initialize(cls,
                   model: M.POSGModel,
                   ego_agent: M.AgentID,
                   gamma: float = 0.9,
                   nesting_level: Union[NestingLevel, List[NestingLevel]] = 1,
                   num_sims: Union[int, List[int], List[List[int]]] = 1000,
                   rollout_policies: Optional[RolloutPolicyMap] = None,
                   uct_c: Optional[float] = None,
                   **kwargs) -> 'NestedSearchTree':
        """Initialize a new Nested Search Tree

        Includes initializing all sub trees.
        """
        if isinstance(nesting_level, list):
            if len(nesting_level) == 1:
                nesting_level = nesting_level[0]
            else:
                nesting_level = nesting_level[ego_agent]

        nested_trees = {}
        if nesting_level > 0:
            for i in range(model.num_agents):
                if i == ego_agent:
                    continue

                nested_tree = cls.initialize(
                    model=model,
                    ego_agent=i,
                    gamma=gamma,
                    nesting_level=nesting_level-1,
                    num_sims=num_sims,
                    rollout_policies=rollout_policies,
                    uct_c=uct_c,
                    **kwargs
                )
                nested_trees[i] = nested_tree

        if isinstance(num_sims, list):
            if len(num_sims) == 1:
                num_sims = num_sims[0]
            else:
                num_sims = num_sims[ego_agent]
        if isinstance(num_sims, list):
            num_sims = num_sims[nesting_level]
        assert isinstance(num_sims, int)

        if rollout_policies is not None:
            pi_cls, pi_kwargs = rollout_policies[ego_agent]
            rollout_policy = pi_cls(model, ego_agent, gamma, **pi_kwargs)
        else:
            rollout_policy = policy.RandomPolicy(
                model, ego_agent, gamma, **kwargs
            )

        if uct_c is None:
            uct_c = abs(model.r_max - model.r_min)

        if kwargs.get("use_reinvigorate_fn", True):
            reinvigorate_fn = model.reinvigorate_fn
        else:
            reinvigorate_fn = None

        ego_nested_tree = cls(
            model=model,
            ego_agent=ego_agent,
            nesting_level=nesting_level,
            nested_trees=nested_trees,
            num_sims=num_sims,
            rollout_policy=rollout_policy,
            gamma=gamma,
            reinvigorate_fn=reinvigorate_fn,
            uct_c=uct_c,
            **kwargs
        )
        return ego_nested_tree
