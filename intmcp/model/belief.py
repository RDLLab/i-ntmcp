"""Belief data structures """
import random
from pprint import pformat
from typing import Optional, Dict, Sequence, List

import numpy as np

from intmcp.model.state import State


class BaseBelief:
    """An abstract belief class """

    def sample(self) -> State:
        """Returns a state from the belief """
        raise NotImplementedError

    def sample_k(self, k: int) -> Sequence[State]:
        """Sample k states from the belief """
        raise NotImplementedError

    def size(self) -> Optional[int]:
        """Returns the size of the belief, or None if not applicable """
        raise NotImplementedError

    def get_dist(self) -> Dict[State, float]:
        """Get belief as a distribution: S -> prob map """
        raise NotImplementedError


class Belief(BaseBelief):
    """A belief class """

    def __init__(self, belief_map: Dict[State, float]):
        self.belief_map = belief_map

    def __call__(self, state: State) -> float:
        """Return the belief probability for given state """
        return self.belief_map.get(state, 0.0)

    def update(self, state: State, prob: float):
        """Set the belief for a given state """
        self.belief_map[state] = prob

    def sample(self) -> State:
        return random.choices(
            self.belief_map,    # type: ignore
            weights=list(self.belief_map.values())
        )[0]

    def sample_k(self, k: int) -> Sequence[State]:
        samples = []
        for _ in range(k):
            samples.append(self.sample())
        return samples

    def size(self) -> Optional[int]:
        return len(self.belief_map)

    def get_dist(self) -> Dict[State, float]:
        """Get belief as a distribution: S -> prob map """
        return self.belief_map


class VectorBelief(Belief):
    """A hasheable vector belief representation """

    # Number of decimals of precision
    DEC_PREC = 12

    def __init__(self,
                 belief_map: Dict[State, float],
                 state_space: Optional[List[State]] = None):
        super().__init__(belief_map)
        if state_space is None:
            self.state_idx_map = {s: i for i, s in enumerate(belief_map)}
        else:
            self.state_idx_map = {s: i for i, s in enumerate(state_space)}

        self.belief_vector = np.zeros(len(self.state_idx_map))
        for s, i in self.state_idx_map.items():
            self.belief_vector[i] = self.belief_map[s]

    def update(self, state: State, prob: float):
        super().update(state, prob)
        self.belief_vector[self.state_idx_map[state]] = prob

    def __eq__(self, other):
        return np.all(np.isclose(self.belief_vector, other.belief_vector))

    def __hash__(self):
        return hash(np.around(self.belief_vector, self.DEC_PREC).tobytes())

    def __repr__(self):
        return pformat(self.belief_map)


class BaseParticleBelief(BaseBelief):
    """An abstract particle belief """

    def sample_k(self, k: int) -> Sequence[State]:
        """Sample k states from the belief """
        raise NotImplementedError

    def extract_k(self, k: int) -> Sequence[State]:
        """Sample up to k states from the belief with no replacement """
        raise NotImplementedError

    def is_depleted(self) -> bool:
        """Returns true if belief is depleted, so cannot be sampled """
        raise NotImplementedError


class InitialParticleBelief(BaseParticleBelief):
    """The initial particle belief for a problem """

    def __init__(self, initial_belief_fn, dist_res: int = 100):
        """Expects initial_belief_fn returns a random initial
        state when called
        """
        self.i_fn = initial_belief_fn
        self.dist_resolution = dist_res

    def sample(self) -> State:
        return self.i_fn()

    def sample_k(self, k: int) -> Sequence[State]:
        samples = []
        for _ in range(k):
            samples.append(self.sample())
        return samples

    def extract_k(self, k: int) -> Sequence[State]:
        return self.sample_k(k)

    def is_depleted(self) -> bool:
        return False

    def size(self) -> Optional[int]:
        return None

    def get_dist(self) -> Dict[State, float]:
        samples = self.sample_k(self.dist_resolution)
        unique_samples = list(set(samples))
        dist = {}
        prob_sum = 0.0
        for state in unique_samples:
            count = samples.count(state)
            prob = count / self.dist_resolution
            dist[state] = prob
            prob_sum += prob

        if prob_sum < 1.0:
            for state, prob in dist.items():
                dist[state] = prob / prob_sum
        return dist


class ParticleBelief(BaseParticleBelief):
    """A belief represented by state particles """

    def __init__(self):
        super().__init__()
        self.particles = []

    def sample(self) -> State:
        return random.choice(self.particles)

    def sample_k(self, k: int) -> Sequence[State]:
        return random.choices(self.particles, k=k)

    def extract_k(self, k: int) -> Sequence[State]:
        if self.size() <= k:
            return self.particles
        return random.choices(self.particles, k=k)

    def add_particle(self, state: State):
        """Add a single state particle to the belief """
        self.particles.append(state)

    def add_particles(self, states: Sequence[State]):
        """Add a multiple state particle to the belief """
        self.particles.extend(states)

    def is_depleted(self) -> bool:
        return len(self.particles) == 0

    def size(self) -> int:
        return len(self.particles)

    def get_dist(self) -> Dict[State, float]:
        """Get the particle distribution for this belief """
        unique_particles = list(set(self.particles))
        dist = {}
        prob_sum = 0.0
        for particle in unique_particles:
            count = self.particles.count(particle)
            prob = count / self.size()
            dist[particle] = prob
            prob_sum += prob

        if prob_sum < 1.0:
            for state, prob in dist.items():
                dist[state] = prob / prob_sum
        return dist
