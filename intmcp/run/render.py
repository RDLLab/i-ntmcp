"""Classes and functions for rendering run graphics """
import abc
from typing import Sequence, Optional, Dict, Any, Iterable, Set, List

import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

import intmcp.model as M
from intmcp import tree as tree_lib
from intmcp import policy as policy_lib


# pylint: disable=[no-self-use]


# Used to map a probability to a color
prob_color_mapper = cm.ScalarMappable(
    matplotlib.colors.Normalize(vmin=0.0, vmax=1.0, clip=True),
    cmap=cm.YlOrRd     # pylint: disable=[no-member]
)


def plot_discrete_dist(ax: matplotlib.axes.Axes,
                       dist: Dict[Any, float]) -> matplotlib.axes.Axes:
    """Plot discrete dist represented by mapping from object to prob.

    Simply returns provided axes.
    """
    labels = [str(x) for x in dist]
    labels.sort()

    label_probs_map = {str(x): p for x, p in dist.items()}
    probs = [label_probs_map.get(x, 0.0) for x in labels]

    x = list(range(len(probs)))
    ax.barh(x, probs, tick_label=labels)
    ax.set_xlim(0.0, 1.0)

    return ax


def get_renderers(show_pi: Optional[int] = None,
                  show_belief: Optional[int] = None,
                  show_tree: Optional[int] = None) -> Sequence['Renderer']:
    """Get set of renderers given flag values """
    renderers: List[Renderer] = []
    if show_pi is not None:
        renderers.append(PolicyRenderer(show_pi))

    if show_belief is not None:
        renderers.append(BeliefRenderer(show_belief))

    if show_tree is not None:
        renderers.append(SearchTreeRenderer(show_tree, None))

    return renderers


class Renderer(abc.ABC):
    """Abstract Renderer Base class """

    FIG_SIZE = (12, 20)

    @abc.abstractmethod
    def render_step(self,
                    env: M.POSGModel,
                    timestep: M.JointTimestep,
                    action: M.JointAction,
                    policies: Sequence[policy_lib.BasePolicy],
                    episode_end: bool) -> None:
        """Render a single environment step """


class SearchTreeRenderer(Renderer):
    """Renders a policies search tree """

    def __init__(self,
                 tree_depth: int,
                 nesting_depth: Optional[int] = None,
                 history_limit: int = 10):
        self._tree_depth = tree_depth
        self._nesting_depth = nesting_depth
        self._history_limit = history_limit

    def render_step(self,
                    env: M.POSGModel,
                    timestep: M.JointTimestep,
                    action: M.JointAction,
                    policies: Sequence[policy_lib.BasePolicy],
                    episode_end: bool) -> None:
        for policy in policies:
            if isinstance(policy, tree_lib.NestedSearchTree):
                self._render_nst(policy)
            # else:
            #     self._render_policy(policy)

    def _render_nst(self, nst: tree_lib.NestedSearchTree) -> None:
        assert nst.num_agents == 2

        nrows = 1
        if self._nesting_depth is None:
            ncols = nst.nesting_level + 1
        else:
            ncols = min(self._nesting_depth, nst.nesting_level) + 1
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)

        histories = {nst.history: 1.0}
        self._render_history_nodes(nst, histories, axs, 0)

        fig.suptitle(f"t={nst.history.t}")
        fig.tight_layout()

    def _render_history_nodes(self,
                              nst: tree_lib.NestedSearchTree,
                              histories: tree_lib.HistoryDist,
                              axs,
                              nesting_depth: int):
        row = 0
        col = nesting_depth
        ax = axs[row][col]

        h_nodes = []
        h_node_colors = []

        history_list = sorted(
            histories.items(), key=lambda x: x[1], reverse=True
        )
        for hist, h_prob in history_list[:self._history_limit]:
            h_nodes.append(nst.traverse(hist))
            h_node_colors.append(prob_color_mapper.to_rgba(h_prob))

        self._plot_nodes(h_nodes, ax, h_node_colors)
        ax.set_title(f"Agent {nst.ego_agent} level={nst.nesting_level}")

        if nst.nesting_level == 0 or nesting_depth == self._nesting_depth:
            return

        nested_histories = nst.get_nested_history_dist(histories)
        for i, nested_tree in nst.nested_trees.items():
            if i == nst.ego_agent:
                continue
            self._render_history_nodes(
                nested_tree, nested_histories[i], axs, nesting_depth + 1
            )

    def _plot_nodes(self,
                    root_nodes: List[tree_lib.Node],
                    ax: matplotlib.axes.Axes,
                    root_node_colors: List[str]):
        """Render a networkx plot of the tree """
        graph = nx.DiGraph()
        for node in root_nodes:
            graph.add_node(node, v=node.value_str(), n=node.n, h=node.h)
            self._recursively_build_tree(graph, node, 0)

        pos = graphviz_layout(graph, prog='dot')
        node_colors = []
        for node in graph.nodes():
            if node in root_nodes:
                color = root_node_colors[root_nodes.index(node)]
            else:
                color = "#A0CBE2"
            node_colors.append(color)
        nx.draw(graph, pos, ax, node_color=node_colors, with_labels=True)

    def _recursively_build_tree(self,
                                graph: nx.DiGraph,
                                parent: tree_lib.Node,
                                depth: int):
        if len(parent.children) == 0 or depth == self._tree_depth:
            return

        for child in parent.children:
            graph.add_node(child, v=child.value_str(), n=child.n, h=child.h)
            graph.add_edge(parent, child)
            self._recursively_build_tree(graph, child, depth+1)


class BeliefRenderer(Renderer):
    """Renders a policies root belief """

    def __init__(self, nesting_depth: Optional[int]):
        self._nesting_depth = nesting_depth

    def render_step(self,
                    env: M.POSGModel,
                    timestep: M.JointTimestep,
                    action: M.JointAction,
                    policies: Sequence[policy_lib.BasePolicy],
                    episode_end: bool) -> None:
        for policy in policies:
            if isinstance(policy, tree_lib.NestedSearchTree):
                self._render_nst(policy)
            # else:
            #     self._render_policy(policy)

    def _render_nst(self, nst: tree_lib.NestedSearchTree) -> None:
        assert nst.num_agents <= 2

        if self._nesting_depth is None or self._nesting_depth == -1:
            ncols = nst.nesting_level + 1
        else:
            ncols = min(nst.nesting_level, self._nesting_depth) + 1

        fig, axs = plt.subplots(
            nrows=1,
            ncols=ncols,
            squeeze=False,
            sharey=True,
            sharex=True
        )

        nested_belief = nst.get_nested_state_beliefs()
        state_set: Set[M.State] = set()
        for agent_belief_map in nested_belief.values():
            for belief in agent_belief_map.values():
                state_set.update(belief)

        nesting_levels = list(nested_belief)
        nesting_levels.sort(reverse=True)

        for depth, nl in enumerate(nesting_levels):
            if depth >= ncols:
                break
            col = depth
            for agent_id, belief in nested_belief[nl].items():
                row = 0
                ax = axs[row][col]

                for s in state_set:
                    if s not in belief:
                        belief[s] = 0.0

                ax = plot_discrete_dist(ax, belief)
                ax.set_title(f"agent={agent_id} level={nl}")

        fig.suptitle(
            f"t={nst.history.t} ego_agent={nst.ego_agent}\n"
            f"Ego history={nst.history}"
        )
        fig.tight_layout()


class PolicyRenderer(Renderer):
    """Renders a policies policy distribution for current root belief """

    def __init__(self, nesting_depth: Optional[int]):
        self._nesting_depth = nesting_depth

    def render_step(self,
                    env: M.POSGModel,
                    timestep: M.JointTimestep,
                    action: M.JointAction,
                    policies: Sequence[policy_lib.BasePolicy],
                    episode_end: bool) -> None:
        for policy in policies:
            if isinstance(policy, tree_lib.NestedSearchTree):
                self._render_nst(policy)
            else:
                self._render_policy(policy)

    def _render_policy(self, policy: policy_lib.BasePolicy) -> None:
        fig = plt.figure()
        ax = fig.subplots(1, 1)

        ego_pi = policy.get_pi_by_history(policy.history)
        ax = plot_discrete_dist(ax, ego_pi)

        fig.suptitle(f"agent={policy.ego_agent} policy={policy}")
        fig.tight_layout()

    def _render_nst(self, nst: tree_lib.NestedSearchTree) -> None:
        if self._nesting_depth is None or self._nesting_depth == -1:
            ncols = nst.nesting_level + 2
        else:
            ncols = min(nst.nesting_level, self._nesting_depth) + 1

        fig = plt.figure()
        axs = fig.subplots(
            nrows=1,
            ncols=ncols,
            squeeze=False,
            sharex=True,
            sharey=False
        )

        nested_pis = nst.get_nested_pis()

        nesting_levels = list(nested_pis)
        nesting_levels.sort(reverse=True)

        for depth, nl in enumerate(nesting_levels):
            if depth >= ncols:
                break
            row = 0
            col = depth
            for agent_id, pi in nested_pis[nl].items():
                ax = axs[row][col]
                ax = plot_discrete_dist(ax, pi)
                ax.set_title(f"level={nl} agent={agent_id}")

        fig.suptitle(f"Agent={nst.ego_agent} policy={nst}")
        fig.tight_layout()


def generate_renders(renderers: Iterable[Renderer],
                     env: M.POSGModel,
                     timestep: M.JointTimestep,
                     action: M.JointAction,
                     policies: Sequence[policy_lib.BasePolicy],
                     episode_end: bool) -> None:
    """Handle the generation of environment step renderings """
    num_renderers = 0
    for renderer in renderers:
        renderer.render_step(env, timestep, action, policies, episode_end)
        num_renderers += 1

    if num_renderers > 0:
        plt.show()
