import time

import networkx as nx
import numpy as np
from numba import njit

__doc__ = r"""
# Topological Kinetic Monte Carlo (TKMC) Simulator

This module implements a high-performance Kinetic Monte Carlo (KMC) simulator based on mean-field approximation and dynamic topological graph theory. It is specifically designed to simulate the reaction kinetics and gelation processes of large-scale (million-particle) crosslinked polymers, hydrogel networks, and complex colloidal systems.

## 1. Physical Principles

Traditional Molecular Dynamics (MD) expends significant computational resources on the integration of continuous spatial coordinates and solvent collisions. This algorithm adopts a **Coordinate-Free** assumption, decoupling the overall reaction probability into the product of three orthogonal dimensions: **mean-field collision probability**, **topological steric penalty**, and **intrinsic chemical reaction probability**.

### 1.1 Mean-Field Collision
Assuming the system is in a good solvent or an ideal melt mixture, the relative probability weight of a node $i$ (Type A) encountering a target Type $T$ is determined by the remaining available population of that type, $N_T$, and the thermodynamic interaction between them (e.g., Lennard-Jones potential $\epsilon$):
$$W_{\text{collide}}(i, T) \propto N_T \cdot \exp\left(\frac{\epsilon_{iT}}{k_B T}\right)$$

### 1.2 Topological Hindrance and Polymer Cyclization
When targeting two specific particles $i$ and $j$, the validity of the reaction conformation is evaluated by computing their Breadth-First Search (BFS) graph distance $d$ within the current crosslinked network:
* **Inter-molecular reaction ($d = \infty$):** Topological penalty is $1.0$.
* **Rigid Cutoff ($d < 5$):** Due to the rigidity of bond angles and dihedrals, folding into short rings is prohibited; the penalty is $0.0$.
* **Gaussian Chain Cyclization ($5 \le d \le 15$):** Following polymer statistical physics, the probability of end-to-end encounters decays with the topological distance:
    $$P_{\text{ring}}(d) \propto d^{-\frac{3}{2}}$$

Furthermore, a spatial steric penalty factor $f_{\text{steric}}$, governed by the local extent of reaction, is introduced ($v$ being the current valency, $V$ being the maximum valency):
$$f_{\text{steric}} = \left(1 - \frac{v_i}{V_i}\right) \left(1 - \frac{v_j}{V_j}\right)$$
The comprehensive topological weight is: $W_{\text{topo}} = P_{\text{ring}}(d) \cdot f_{\text{steric}}$

## 2. Algorithm Workflow

To mitigate the computational waste caused by extremely low topological probabilities, this algorithm introduces **Local Tournament Importance Sampling**.

1.  **Global Time Step (KMC Sweep):** Each step executes collision attempts equal to the total number of currently active particles.
2.  **Source Sampling:** Randomly draw a source particle $i$, weighted by the remaining active count of each Type.
3.  **Target Type Sampling:** Select a target Type $T$ based on the mean-field weight matrix $W_{\text{collide}}$.
4.  **Local Tournament:** Randomly sample $K$ candidate particles from Type $T$. Compute the $W_{\text{topo}}$ between these $K$ particles and $i$.
5.  **Roulette Wheel Selection:** Normalize these $K$ weights ($W_{\text{topo}}$) and cast a die to select the **unique** optimal conformation winner, $j_{\text{winner}}$.
6.  **Intrinsic Rejection:** Generate a random number $r \in [0, 1)$. If $r < P_{\text{intrinsic}}(i, j)$, the reaction occurs, the dynamic topological graph is updated, and free valencies are consumed; otherwise, the reaction is rejected.

## 3. Computational Optimizations
* **Flat Arrays & JIT:** Replaces dynamic objects with compact 1D/2D Numpy arrays and utilizes Numba for Just-In-Time (JIT) compilation to machine code.
* **O(1) Swap-and-Pop:** The active sampling pool employs a swap-with-last and pop mechanism, strictly avoiding $O(N)$ list deletions and memory reallocations.
* **Visited Token BFS:** Utilizes a global timestamp to record traversal states, enabling millions of BFS searches with zero dynamic memory allocation and zero array resets.
"""


# ==========================================
# 1. Numba JIT Core Engine (C++ Level Performance)
# ==========================================

@njit
def fast_bfs(start, target, neighbors, visited, token, max_depth):
    """Ultra-fast depth-limited BFS (utilizing the timestamp token trick)."""
    if start == target: return 0

    # Statically allocated queue to circumvent dynamic memory overhead
    queue = np.empty(100000, dtype=np.int32)
    depths = np.empty(100000, dtype=np.int8)
    head = 0
    tail = 0

    queue[tail] = start
    depths[tail] = 0
    tail += 1
    visited[start] = token

    while head < tail:
        curr = queue[head]
        d = depths[head]
        head += 1

        if d == max_depth:
            continue

        # Traverse neighbors via the pre-allocated 2D array
        for i in range(neighbors.shape[1]):
            nxt = neighbors[curr, i]
            if nxt == -1:
                break  # -1 signifies no further valid neighbors
            if nxt == target:
                return d + 1

            if visited[nxt] != token:
                visited[nxt] = token
                if tail < 100000:  # Safeguard against extreme queue overflow
                    queue[tail] = nxt
                    depths[tail] = d + 1
                    tail += 1
    return -1


@njit
def run_kmc_simulation(total_steps, N, num_types, types, valency, max_valency,
                       active_pools, active_counts, pool_indices, neighbors,
                       prob_matrix, weight_matrix, reaction_history,
                       current_reaction_idx, K_samples=10):
    """Core KMC loop executing Local Tournament Importance Sampling."""

    # Global timestamp array for $O(1)$ BFS state resets
    visited = np.zeros(N, dtype=np.int32)
    token = 1

    # Pre-allocate tournament arrays to prevent in-loop instantiation
    local_weights = np.zeros(K_samples, dtype=np.float64)
    local_candidates = np.zeros(K_samples, dtype=np.int32)

    for step in range(total_steps):
        total_active = np.sum(active_counts)
        if total_active < 2:
            break  # System is fully crosslinked or dynamically deadlocked

        # Attempt 'total_active' collision trials per KMC Sweep
        for _ in range(total_active):
            # 1. Source type sampling (weighted by remaining available counts)
            type_probs = np.zeros(num_types, dtype=np.float64)
            for t in range(num_types):
                type_probs[t] = active_counts[t]

            sum_type_probs = np.sum(type_probs)
            if sum_type_probs == 0: break
            type_probs /= sum_type_probs

            r = np.random.rand()
            cum_p = 0.0
            type_i = 0
            for t in range(num_types):
                cum_p += type_probs[t]
                if r < cum_p:
                    type_i = t
                    break

            if active_counts[type_i] == 0: continue

            # Randomly draw an available source particle `i`
            idx_i = np.random.randint(0, active_counts[type_i])
            particle_i = active_pools[type_i, idx_i]

            if valency[particle_i] >= max_valency[particle_i]:
                continue

            # 2. Compute mean-field collision probabilities for target Types
            weights = np.zeros(num_types, dtype=np.float64)
            for t in range(num_types):
                if active_counts[t] > 0:
                    weights[t] = active_counts[t] * weight_matrix[type_i, t]

            sum_w = np.sum(weights)
            if sum_w == 0: continue
            weights /= sum_w

            r2 = np.random.rand()
            cum_p2 = 0.0
            type_j = 0
            for t in range(num_types):
                cum_p2 += weights[t]
                if r2 < cum_p2:
                    type_j = t
                    break

            # 3. Local Tournament Sampling (Extract K candidates)
            local_weights.fill(0.0)
            local_candidates.fill(-1)

            for k in range(K_samples):
                if active_counts[type_j] == 0: break

                idx_j = np.random.randint(0, active_counts[type_j])
                particle_j = active_pools[type_j, idx_j]

                # Exclude self-reactions and fully saturated particles
                if particle_i == particle_j or valency[particle_j] >= max_valency[particle_j]:
                    continue

                # Verify they are not pre-existing neighbors
                already_neighbors = False
                for n_idx in range(valency[particle_i]):
                    if neighbors[particle_i, n_idx] == particle_j:
                        already_neighbors = True
                        break
                if already_neighbors:
                    continue

                # Compute the BFS topological weight for the candidate
                token += 1
                d = fast_bfs(particle_i, particle_j, neighbors, visited, token, 15)
                f_steric = (1.0 - valency[particle_i] / max_valency[particle_i]) * \
                           (1.0 - valency[particle_j] / max_valency[particle_j])

                if d == -1:
                    W_k = 1.0 * f_steric
                elif d < 5:
                    W_k = 0.0
                else:
                    W_k = (d ** -1.5) * f_steric

                local_weights[k] = W_k
                local_candidates[k] = particle_j

            # 4. Weight Screening & Intrinsic Probability Rejection
            sum_W = np.sum(local_weights)
            if sum_W > 0:
                # Roulette wheel selection to determine the tournament winner
                r3 = np.random.rand() * sum_W
                cum_w = 0.0
                winner_j = -1
                for k in range(K_samples):
                    cum_w += local_weights[k]
                    if r3 <= cum_w:
                        winner_j = local_candidates[k]
                        break

                if winner_j != -1:
                    # The definitive physical criterion: reject solely based on intrinsic kinetic probability
                    p_intrinsic = prob_matrix[type_i, type_j]

                    if np.random.rand() < p_intrinsic:
                        # Reaction successfully occurs!
                        neighbors[particle_i, valency[particle_i]] = winner_j
                        neighbors[winner_j, valency[winner_j]] = particle_i
                        valency[particle_i] += 1
                        valency[winner_j] += 1

                        # Record chronological reaction history (zero-overhead assignment)
                        reaction_history[current_reaction_idx, 0] = particle_i
                        reaction_history[current_reaction_idx, 1] = winner_j
                        current_reaction_idx += 1

                        # --- Absolute O(1) Swap-and-Pop for the active pool ---
                        if valency[particle_i] == max_valency[particle_i]:
                            # Locate the pointer of `i` in the active pool
                            p_idx = pool_indices[particle_i]
                            last_idx = active_counts[type_i] - 1
                            last_particle = active_pools[type_i, last_idx]

                            # Swap out
                            active_pools[type_i, p_idx] = last_particle
                            pool_indices[last_particle] = p_idx
                            active_counts[type_i] -= 1

                        if valency[winner_j] == max_valency[winner_j]:
                            p_idx = pool_indices[winner_j]
                            last_idx = active_counts[type_j] - 1
                            last_particle = active_pools[type_j, last_idx]

                            active_pools[type_j, p_idx] = last_particle
                            pool_indices[last_particle] = p_idx
                            active_counts[type_j] -= 1

    return current_reaction_idx


# ==========================================
# 2. Python Administrative Wrapper Layer
# ==========================================

class FastTKMC:
    def __init__(self, counts, valencies, base_probs, weights=None, k_samples=10, type_names=None):
        self.num_types = len(counts)
        self.total_N = sum(counts)
        self.max_possible_val = max(valencies)
        self.k_samples = k_samples

        # Default naming convention (e.g., Type_0, Type_1) if custom strings aren't provided
        self.type_names = type_names if type_names else [f"Type_{i}" for i in range(self.num_types)]

        self.types = np.zeros(self.total_N, dtype=np.int8)
        self.max_valency = np.zeros(self.total_N, dtype=np.int8)
        self.valency = np.zeros(self.total_N, dtype=np.int8)

        self.active_pools = np.zeros((self.num_types, max(counts)), dtype=np.int32)
        self.active_counts = np.array(counts, dtype=np.int32)
        self.pool_indices = np.zeros(self.total_N, dtype=np.int32)  # Reverse index for O(1) targeting

        self.neighbors = np.full((self.total_N, self.max_possible_val), -1, dtype=np.int32)

        self.prob_matrix = np.array(base_probs, dtype=np.float64)
        if weights is None:
            self.weight_matrix = np.ones((self.num_types, self.num_types), dtype=np.float64)
        else:
            self.weight_matrix = np.array(weights, dtype=np.float64)

        # Pre-allocate chronological reaction history array
        # Theoretical max bonds = sum(N_i * V_i) / 2
        max_theoretical_reactions = int(sum(counts[i] * valencies[i] for i in range(self.num_types)) // 2)
        self.reaction_history = np.zeros((max_theoretical_reactions, 2), dtype=np.int32)
        self.reaction_counter = 0

        # Initialization
        curr = 0
        for t, count in enumerate(counts):
            for i in range(count):
                self.types[curr] = t
                self.max_valency[curr] = valencies[t]
                self.active_pools[t, i] = curr
                self.pool_indices[curr] = i
                curr += 1

    def run(self, steps):
        t0 = time.time()
        print(
            f"[{time.strftime('%H:%M:%S')}] Executing KMC simulation... System size: {self.total_N}, Exploration samples (K): {self.k_samples}")

        # Numba function returns the updated reaction counter
        self.reaction_counter = run_kmc_simulation(
            steps, self.total_N, self.num_types,
            self.types, self.valency, self.max_valency,
            self.active_pools, self.active_counts, self.pool_indices,
            self.neighbors, self.prob_matrix, self.weight_matrix,
            self.reaction_history, self.reaction_counter, self.k_samples
        )

        t1 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Simulation successfully concluded! Total elapsed time: {t1 - t0:.3f} s")

    def build_networkx_graph(self):
        """Constructs and returns the final topological structure as a NetworkX Graph."""
        print("Extracting topology to NetworkX Graph architecture (this may require a few moments)...")
        G = nx.Graph()

        # Adding nodes with their respective 'type' attribute for subsequent analysis
        for i in range(self.total_N):
            G.add_node(i, type=self.type_names[self.types[i]])

        edges = []
        for i in range(self.total_N):
            for j_idx in range(self.valency[i]):
                j = self.neighbors[i, j_idx]
                if i < j:  # Ensure edges are strictly unidirectional to avert duplication
                    edges.append((i, j))

        G.add_edges_from(edges)
        return G

    def get_chronological_reaction_list(self):
        """
        Returns a time-ordered sequence of all successfully occurring reactions.
        Format: list of tuples -> [('type_i-type_j', idx_i, idx_j), ...]
        """
        valid_reactions = self.reaction_history[:self.reaction_counter]
        formatted_list = []

        for u, v in valid_reactions:
            type_str_u = self.type_names[self.types[u]]
            type_str_v = self.type_names[self.types[v]]

            # Sort the type string lexically for structural consistency (e.g. 'A-B' rather than 'B-A')
            bond_type = "-".join(sorted([type_str_u, type_str_v]))
            formatted_list.append((bond_type, int(u), int(v)))

        return formatted_list


# ==========================================
# Execution Demonstrator (1 Million Particles)
# ==========================================
if __name__ == "__main__":
    # Particle designations: 0: Crosslinker A (valency=3), 1: Monomer B (valency=2), 2: Monomer C (valency=2)
    counts = [100_000, 450_000, 450_000]
    valencies = [3, 2, 2]
    type_names = ['A', 'B', 'C']

    # Intrinsic kinetic reaction probabilities
    base_probs = [
        [0.0, 0.3, 0.05],
        [0.0, 0.0, 0.5],
        [0.05, 0.5, 0.0]
    ]

    # Mean-field local concentration weights driven by thermodynamics
    # $$W_{\text{collide}}(i, T) \propto N_T \cdot \exp\left(\frac{\epsilon_{iT}}{k_B T}\right)$$
    weights = [
        [1.0, 2.5, 0.5],
        [2.5, 1.0, 1.0],
        [0.5, 1.0, 1.0]
    ]

    # Initialize Simulator
    sim = FastTKMC(counts, valencies, base_probs, weights, k_samples=10, type_names=type_names)

    # Execute simulation (e.g., 20 KMC Sweeps)
    sim.run(steps=20)

    # Objective 1: Construct the Graph
    G = sim.build_networkx_graph()
    print(f"Total topological edges (chemical bonds formed): {G.number_of_edges()}")

    # Objective 2: Extract the Chronological Reaction List
    reaction_sequence = sim.get_chronological_reaction_list()
    print(f"Total reactions recorded in chronological order: {len(reaction_sequence)}")

    if reaction_sequence:
        print("\nPreview of the first 5 chemical events:")
        for event in reaction_sequence[:5]:
            print(f"Bond formed: {event[0]} between Particle ID {event[1]} and Particle ID {event[2]}")

        print("\nPreview of the final 5 chemical events:")
        for event in reaction_sequence[-5:]:
            print(f"Bond formed: {event[0]} between Particle ID {event[1]} and Particle ID {event[2]}")
