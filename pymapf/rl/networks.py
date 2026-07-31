"""Actor-critic networks, with a numpy reference and an optional torch backend.

Two backends, for the same reason the solvers have no runtime dependencies: the
numpy one runs anywhere this library runs, including a CI job with nothing
installed, so the learning code is *tested* rather than merely shipped. It is a
small MLP with hand-written backpropagation and Adam -- about as much machinery
as PPO on a grid actually needs, and gradient-checked against finite differences
in the test suite, which is the only honest way to claim hand-derived gradients
are right.

The torch backend exists because the numpy one will not take you past small
networks. It is selected by name and implements the same three methods, so
nothing above it changes::

    make_actor_critic("numpy", obs_dim, n_actions, critic_dim)
    make_actor_critic("torch", obs_dim, n_actions, critic_dim)

The interface is deliberately narrow -- ``act``, ``evaluate``, ``update`` -- so
a third backend is a small, self-contained thing to write.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "ActorCritic",
    "NumpyActorCritic",
    "TorchActorCritic",
    "make_actor_critic",
    "available_backends",
    "MLP",
    "Adam",
]


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


class Adam:
    """Adam, on a list of parameter arrays (Kingma and Ba, 2015)."""

    def __init__(self, shapes: Sequence[Tuple[int, ...]], lr: float = 3e-4,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = [np.zeros(shape) for shape in shapes]
        self.v = [np.zeros(shape) for shape in shapes]
        self.t = 0

    def step(self, params: List[np.ndarray], grads: List[np.ndarray]) -> None:
        self.t += 1
        correction1 = 1 - self.beta1 ** self.t
        correction2 = 1 - self.beta2 ** self.t
        for index, (param, grad) in enumerate(zip(params, grads)):
            self.m[index] = self.beta1 * self.m[index] + (1 - self.beta1) * grad
            self.v[index] = self.beta2 * self.v[index] + (1 - self.beta2) * grad * grad
            m_hat = self.m[index] / correction1
            v_hat = self.v[index] / correction2
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class MLP:
    """A tanh MLP with explicit forward and backward passes.

    Orthogonal initialisation with the scaling PPO implementations conventionally
    use -- ``sqrt(2)`` on hidden layers, and a small gain on the output head so
    the initial policy is close to uniform and the initial value estimates are
    close to zero. Both matter more than they look: a policy that starts
    confident spends its first updates undoing that confidence.
    """

    def __init__(self, sizes: Sequence[int], output_gain: float = 0.01, seed: Optional[int] = None):
        self.sizes = list(sizes)
        random = np.random.default_rng(seed)
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for index in range(len(sizes) - 1):
            gain = np.sqrt(2.0) if index < len(sizes) - 2 else output_gain
            self.weights.append(_orthogonal((sizes[index], sizes[index + 1]), gain, random))
            self.biases.append(np.zeros(sizes[index + 1]))

    @property
    def params(self) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for weight, bias in zip(self.weights, self.biases):
            out.extend([weight, bias])
        return out

    @property
    def shapes(self) -> List[Tuple[int, ...]]:
        return [param.shape for param in self.params]

    def forward(self, x: np.ndarray):
        """Returns the output and the activations needed by :meth:`backward`."""
        activations = [x]
        current = x
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            current = current @ weight + bias
            if index < len(self.weights) - 1:
                current = np.tanh(current)
                activations.append(current)
        return current, activations

    def backward(self, grad_output: np.ndarray, activations: List[np.ndarray]) -> List[np.ndarray]:
        """Gradients w.r.t. the parameters, given dL/d(output)."""
        grads: List[np.ndarray] = [None] * (2 * len(self.weights))
        delta = grad_output
        for index in reversed(range(len(self.weights))):
            activation = activations[index]
            grads[2 * index] = activation.T @ delta
            grads[2 * index + 1] = delta.sum(axis=0)
            if index > 0:
                delta = (delta @ self.weights[index].T) * (1 - activation ** 2)
        return grads


def _orthogonal(shape: Tuple[int, int], gain: float, random) -> np.ndarray:
    """Orthogonal initialisation for any shape, square or not.

    ``np.linalg.qr`` returns the *reduced* factorisation, so for a wide matrix
    it hands back a square ``q`` of the wrong size. Factorising the tall
    orientation and transposing afterwards is the standard fix, and the one
    ``torch.nn.init.orthogonal_`` uses -- without it the value network silently
    gets a square weight matrix and the first forward pass fails to broadcast.
    """
    rows, cols = shape
    flat = random.normal(size=(max(rows, cols), min(rows, cols)))
    q, r = np.linalg.qr(flat)
    # Sign-correct the columns so the result is drawn uniformly from O(n).
    q = q * np.sign(np.diag(r))
    if rows < cols:
        q = q.T
    # Forced contiguous, and not as a micro-optimisation. Multiplying a
    # transposed array keeps its Fortran order, and a non-contiguous parameter
    # makes `param.reshape(-1)` return a *copy* -- so anything that writes
    # through that view (a finite-difference gradient check, most obviously)
    # silently updates nothing and reports a zero gradient.
    return np.ascontiguousarray(gain * q[:rows, :cols])


class ActorCritic(ABC):
    """What a PPO trainer needs from a network, and nothing more."""

    @abstractmethod
    def act(self, observations: np.ndarray, deterministic: bool = False):
        """Sample actions. Returns ``(actions, log_probs)``."""

    @abstractmethod
    def value(self, critic_inputs: np.ndarray) -> np.ndarray:
        """State values. The input is local observations for IPPO, the global
        state for MAPPO -- which is the entire difference between them."""

    @abstractmethod
    def update(
        self,
        batch: Dict[str, np.ndarray],
        clip: float = 0.2,
        value_coefficient: float = 0.5,
        entropy_coefficient: float = 0.01,
        max_grad_norm: float = 0.5,
    ) -> Dict[str, float]:
        """One PPO step on a minibatch. Returns diagnostics.

        The hyperparameters are named rather than collected into ``**kwargs``.
        An open-ended signature here would promise callers something no backend
        actually honours -- every implementation takes exactly these four, and
        declaring otherwise is a Liskov violation that a linter is right to
        flag.
        """

    def state_dict(self) -> dict:
        raise NotImplementedError

    def load_state_dict(self, state: dict) -> None:
        raise NotImplementedError


class NumpyActorCritic(ActorCritic):
    """Reference implementation: two MLPs, hand-written gradients, Adam.

    Separate actor and critic trunks rather than a shared one. Sharing saves
    parameters and couples the two losses, and the coupling is what makes PPO
    implementations sensitive to the value-loss coefficient; keeping them apart
    is the boring choice that behaves.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        critic_dim: Optional[int] = None,
        hidden: Sequence[int] = (64, 64),
        lr: float = 3e-4,
        seed: Optional[int] = None,
    ):
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.critic_dim = int(critic_dim if critic_dim is not None else obs_dim)
        self.actor = MLP([self.obs_dim, *hidden, self.n_actions], output_gain=0.01, seed=seed)
        self.critic = MLP(
            [self.critic_dim, *hidden, 1],
            output_gain=1.0,
            seed=None if seed is None else seed + 1,
        )
        self._random = np.random.default_rng(seed)
        self.actor_optimiser = Adam(self.actor.shapes, lr=lr)
        self.critic_optimiser = Adam(self.critic.shapes, lr=lr)

    # -- inference ----------------------------------------------------
    def logits(self, observations: np.ndarray) -> np.ndarray:
        return self.actor.forward(np.atleast_2d(observations))[0]

    def act(self, observations: np.ndarray, deterministic: bool = False):
        logits = self.logits(observations)
        log_probs = _log_softmax(logits)
        if deterministic:
            actions = np.argmax(log_probs, axis=-1)
        else:
            probabilities = np.exp(log_probs)
            cumulative = probabilities.cumsum(axis=-1)
            draws = self._random.random((probabilities.shape[0], 1))
            actions = (draws < cumulative).argmax(axis=-1)
        chosen = log_probs[np.arange(len(actions)), actions]
        return actions, chosen

    def value(self, critic_inputs: np.ndarray) -> np.ndarray:
        return self.critic.forward(np.atleast_2d(critic_inputs))[0].reshape(-1)

    # -- learning -----------------------------------------------------
    def update(
        self,
        batch: Dict[str, np.ndarray],
        clip: float = 0.2,
        value_coefficient: float = 0.5,
        entropy_coefficient: float = 0.01,
        max_grad_norm: float = 0.5,
    ) -> Dict[str, float]:
        observations = batch["observations"]
        critic_inputs = batch["critic_inputs"]
        actions = batch["actions"].astype(int)
        old_log_probs = batch["log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        count = len(actions)
        rows = np.arange(count)

        # ---- actor -------------------------------------------------
        logits, activations = self.actor.forward(observations)
        log_probs = _log_softmax(logits)
        probabilities = np.exp(log_probs)
        chosen = log_probs[rows, actions]
        ratio = np.exp(chosen - old_log_probs)

        unclipped = ratio * advantages
        clipped = np.clip(ratio, 1 - clip, 1 + clip) * advantages
        # PPO maximises the minimum of the two, so the loss is its negation.
        use_unclipped = unclipped <= clipped
        d_ratio = np.where(use_unclipped, -advantages / count, 0.0)

        # d(log pi_a)/d(logits) = onehot(a) - softmax(logits)
        onehot = np.zeros_like(logits)
        onehot[rows, actions] = 1.0
        d_logits = (d_ratio * ratio)[:, None] * (onehot - probabilities)

        entropy = -(probabilities * log_probs).sum(axis=-1)
        # d(entropy)/d(logits) for a softmax, summed and averaged over the batch.
        d_entropy = probabilities * (log_probs + entropy[:, None])
        d_logits += entropy_coefficient * d_entropy / count

        actor_grads = self.actor.backward(d_logits, activations)
        _clip_global_norm(actor_grads, max_grad_norm)
        self.actor_optimiser.step(self.actor.params, actor_grads)

        # ---- critic ------------------------------------------------
        values, critic_activations = self.critic.forward(critic_inputs)
        values = values.reshape(-1)
        residual = values - returns
        d_values = (value_coefficient * residual / count).reshape(-1, 1)
        critic_grads = self.critic.backward(d_values, critic_activations)
        _clip_global_norm(critic_grads, max_grad_norm)
        self.critic_optimiser.step(self.critic.params, critic_grads)

        with np.errstate(over="ignore"):
            approx_kl = float(np.mean(old_log_probs - chosen))
        return {
            "policy_loss": float(-np.mean(np.minimum(unclipped, clipped))),
            "value_loss": float(0.5 * np.mean(residual ** 2)),
            "entropy": float(np.mean(entropy)),
            "approx_kl": approx_kl,
            "clip_fraction": float(np.mean(np.abs(ratio - 1) > clip)),
        }

    def state_dict(self) -> dict:
        return {
            "actor": [param.copy() for param in self.actor.params],
            "critic": [param.copy() for param in self.critic.params],
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "critic_dim": self.critic_dim,
        }

    def load_state_dict(self, state: dict) -> None:
        for param, saved in zip(self.actor.params, state["actor"]):
            param[...] = saved
        for param, saved in zip(self.critic.params, state["critic"]):
            param[...] = saved


def _clip_global_norm(grads: List[np.ndarray], max_norm: float) -> float:
    """Scale gradients so their global L2 norm is at most ``max_norm``."""
    if not max_norm:
        return 0.0
    total = np.sqrt(sum(float((grad ** 2).sum()) for grad in grads))
    if total > max_norm:
        scale = max_norm / (total + 1e-8)
        for grad in grads:
            grad *= scale
    return total


class TorchActorCritic(ActorCritic):
    """The same interface on torch, for when the numpy backend runs out.

    Kept deliberately thin: identical architecture, identical PPO objective, so
    a discrepancy between the two backends is a bug in one of them rather than
    a difference of opinion. The test suite runs the same learning check against
    whichever backends are installed.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        critic_dim: Optional[int] = None,
        hidden: Sequence[int] = (64, 64),
        lr: float = 3e-4,
        seed: Optional[int] = None,
        device: str = "cpu",
    ):
        try:
            import torch
            from torch import nn
        except ImportError as error:  # pragma: no cover - needs torch
            raise ImportError(
                "torch is not installed; `pip install pymapf[rl-torch]`, or use "
                "backend='numpy', which needs nothing beyond numpy"
            ) from error

        self._torch = torch
        if seed is not None:
            torch.manual_seed(seed)
        self.device = torch.device(device)
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.critic_dim = int(critic_dim if critic_dim is not None else obs_dim)

        def build(sizes, output_gain):
            layers = []
            for index in range(len(sizes) - 1):
                linear = nn.Linear(sizes[index], sizes[index + 1])
                gain = np.sqrt(2.0) if index < len(sizes) - 2 else output_gain
                nn.init.orthogonal_(linear.weight, gain)
                nn.init.zeros_(linear.bias)
                layers.append(linear)
                if index < len(sizes) - 2:
                    layers.append(nn.Tanh())
            return nn.Sequential(*layers)

        self.actor = build([self.obs_dim, *hidden, self.n_actions], 0.01).to(self.device)
        self.critic = build([self.critic_dim, *hidden, 1], 1.0).to(self.device)
        self.optimiser = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

    def _tensor(self, array):
        return self._torch.as_tensor(
            np.asarray(array, dtype=np.float32), device=self.device
        )

    def act(self, observations: np.ndarray, deterministic: bool = False):
        torch = self._torch
        with torch.no_grad():
            logits = self.actor(self._tensor(np.atleast_2d(observations)))
            distribution = torch.distributions.Categorical(logits=logits)
            actions = logits.argmax(-1) if deterministic else distribution.sample()
            log_probs = distribution.log_prob(actions)
        return actions.cpu().numpy(), log_probs.cpu().numpy()

    def value(self, critic_inputs: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            return self.critic(self._tensor(np.atleast_2d(critic_inputs))).squeeze(-1).cpu().numpy()

    def update(
        self,
        batch: Dict[str, np.ndarray],
        clip: float = 0.2,
        value_coefficient: float = 0.5,
        entropy_coefficient: float = 0.01,
        max_grad_norm: float = 0.5,
    ) -> Dict[str, float]:
        torch = self._torch
        observations = self._tensor(batch["observations"])
        critic_inputs = self._tensor(batch["critic_inputs"])
        actions = torch.as_tensor(batch["actions"].astype(np.int64), device=self.device)
        old_log_probs = self._tensor(batch["log_probs"])
        advantages = self._tensor(batch["advantages"])
        returns = self._tensor(batch["returns"])

        logits = self.actor(observations)
        distribution = torch.distributions.Categorical(logits=logits)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy().mean()

        ratio = torch.exp(log_probs - old_log_probs)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages
        policy_loss = -torch.min(unclipped, clipped).mean()

        values = self.critic(critic_inputs).squeeze(-1)
        value_loss = 0.5 * ((values - returns) ** 2).mean()

        loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy
        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()), max_grad_norm
        )
        self.optimiser.step()

        with torch.no_grad():
            approx_kl = float((old_log_probs - log_probs).mean())
            clip_fraction = float(((ratio - 1).abs() > clip).float().mean())
        return {
            # Detached before the conversion: reading a scalar straight off a
            # tensor that still needs grad works, but warns, and the warning is
            # right that it is not what you meant.
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "entropy": float(entropy.detach()),
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
        }

    def state_dict(self) -> dict:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])


BACKENDS = {"numpy": NumpyActorCritic, "torch": TorchActorCritic}


def available_backends() -> List[str]:
    """Backend names that can actually be constructed on this machine."""
    usable = ["numpy"]
    try:  # pragma: no cover - depends on the environment
        # Imported purely to find out whether it is importable.
        import torch  # noqa: F401  # pylint: disable=unused-import

        usable.append("torch")
    except ImportError:
        pass
    return usable


def make_actor_critic(backend: str, *args, **kwargs) -> ActorCritic:
    try:
        factory = BACKENDS[str(backend).lower()]
    except KeyError as error:
        raise ValueError(
            "Unknown backend %r. Available: %s" % (backend, ", ".join(sorted(BACKENDS)))
        ) from error
    return factory(*args, **kwargs)
