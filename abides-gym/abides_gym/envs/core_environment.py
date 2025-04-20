from copy import deepcopy
from abc import abstractmethod, ABC
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding
import os
import torch as th
from stable_baselines3 import PPO

from abides_core import Kernel, NanosecondTime
from abides_core.generators import InterArrivalTimeGenerator
from abides_core.utils import subdict
from abides_markets.utils import config_add_agents
from abides_gym.experimental_agents import SelfPlayAgent


class AbidesGymCoreEnv(gym.Env, ABC):
    """
    Abstract class for core gym to inherit from to create usable specific ABIDES Gyms
    """

    def __init__(
            self,
            background_config_pair: Tuple[Callable | List[Callable], Optional[Dict[str, Any]]],
            wakeup_interval_generator: InterArrivalTimeGenerator,
            state_buffer_length: int,
            first_interval: Optional[NanosecondTime] = None,
            gymAgentConstructor=None,
            saved_models_location=None
    ) -> None:

        self.background_config_pair: Tuple[
            Callable | List[Callable], Optional[Dict[str, Any]]
        ] = background_config_pair
        if background_config_pair[1] is None:
            background_config_pair[1] = {}

        self.wakeup_interval_generator: InterArrivalTimeGenerator = (
            wakeup_interval_generator
        )
        self.first_interval = first_interval
        self.state_buffer_length: int = state_buffer_length
        self.gymAgentConstructor = gymAgentConstructor

        self.seed()  # fix random seed if no seed specified

        self.state: Optional[np.ndarray] = None
        self.reward: Optional[float] = None
        self.truncated: Optional[bool] = None
        self.terminated: Optional[bool] = None
        self.info: Optional[Dict[str, Any]] = None
        self.self_play_agents: List[SelfPlayAgent] = None
        self.saved_models_location: str = saved_models_location
        self.environment_configuration: Dict[str, Any] = {}

    def set_environment_configuration(self, env_config: Dict[str, Any]):
        self.environment_configuration = env_config

    def reset(self,
              *,
              seed: int | None = None,
              options: dict[str, Any] | None = None,
              ):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """

        # get seed to initialize random states for ABIDES
        seed = self.np_random.integers(low=0, high=2 ** 32, dtype="uint64") if not seed else seed
        # instanciate back ground config state
        background_config_args = self.background_config_pair[1]
        background_config_args.update(
            {"seed": seed, **self.extra_background_config_kvargs}
        )

        if isinstance(self.background_config_pair[0], List):
            random_env = np.random.choice(self.background_config_pair[0])
            background_config_state = random_env(**background_config_args)
        else:
            background_config_state = self.background_config_pair[0](
                **background_config_args
            )

        self_play_agents = []
        agents = []
        if background_config_args.get('n_self_play_agents', 0) > 0 and self.saved_models_location:
            n = len(background_config_state["agents"])
            for i in range(background_config_args['n_self_play_agents']):
                # 1. Get the random network from the saved_models
                #   - Where are we going to store this?
                #   - Has to be accessible in a central spot and since it will not be trained here we need a common PWD
                # Get the location of the models
                models_dir = self.saved_models_location
                # Get the models in the directory
                options = os.listdir(models_dir)
                # Choose any of the models
                model_chosen = np.random.choice(options)

                model = PPO.load(models_dir + '/' + model_chosen)
                # 2. Make sure the correct configuration is given along
                # 3. Make sure the model is unzipped and given as is to the network as a module
                # 4. Continue executiing

                new_sp_agent = SelfPlayAgent(
                    n + i,
                    "ABM",
                    first_interval=self.first_interval,
                    wakeup_interval_generator=self.wakeup_interval_generator,
                    state_buffer_length=self.state_buffer_length,
                    nn_model=model,
                    environment_configuration=self.environment_configuration,
                    **self.extra_gym_agent_kvargs
                )
                self_play_agents.append(new_sp_agent)
        # instantiate gym agent and add it to config and gym object
        nextid = len(background_config_state["agents"]) + background_config_args.get('n_self_play_agents', 0)
        gym_agent = self.gymAgentConstructor(
            nextid,
            "ABM",
            first_interval=self.first_interval,
            wakeup_interval_generator=self.wakeup_interval_generator,
            state_buffer_length=self.state_buffer_length,
            **self.extra_gym_agent_kvargs,
        )

        agents.extend(self_play_agents) if len(self_play_agents) > 0 else agents
        agents.append(gym_agent)
        config_state = config_add_agents(background_config_state, agents)

        self.gym_agent = config_state["agents"][-1]
        # KERNEL
        # instantiate the kernel object
        kernel = Kernel(
            random_state=np.random.RandomState(seed=seed),
            **subdict(
                config_state,
                [
                    "start_time",
                    "stop_time",
                    "agents",
                    "agent_latency_model",
                    "default_computation_delay",
                    "custom_properties",
                ],
            ),
        )
        kernel.initialize()
        # kernel will run until GymAgent has to take an action
        raw_state = kernel.runner()

        # Now we also want to trigger the remaining agents in the environment with their raw_state and information

        state = self.raw_state_to_state(deepcopy(raw_state["result"]))
        info = self.raw_state_to_info(deepcopy(raw_state["result"]))

        # Make sure the SP agents have the correct raw_state at this very instant
        for sp_agent in self_play_agents:
            sp_agent.update_raw_state()

        self.self_play_agents = self_play_agents
        # attach kernel
        self.kernel = kernel
        return state, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : Discrete

        Returns
        -------
        observation, reward, done, info : tuple
            observation (object) :
                an environment-specific object representing your observation of
                the environment.

            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.

            terminated (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)

            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        assert self.action_space.contains(
            action
        ), f"Action {action} is not contained in Action Space"

        abides_action = self._map_action_space_to_ABIDES_SIMULATOR_SPACE(action)

        raw_state = self.kernel.runner((self.gym_agent, abides_action))
        self.state = self.raw_state_to_state(deepcopy(raw_state["result"]))

        assert self.observation_space.contains(
            self.state
        ), f"INVALID STATE {self.state}"

        self.reward = self.raw_state_to_reward(deepcopy(raw_state["result"]))
        self.done = raw_state["done"] or self.raw_state_to_done(
            deepcopy(raw_state["result"])
        )

        if self.done:
            self.reward += self.raw_state_to_update_reward(
                deepcopy(raw_state["result"])
            )

        self.info = self.raw_state_to_info(deepcopy(raw_state["result"]))

        return (self.state, self.reward, self.done, False, self.info)

    def render(self, mode: str = "human") -> None:
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with
        """
        print(self.state, self.reward, self.info)

    def seed(self, seed: Optional[int] = None) -> List[Any]:
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self) -> None:
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        # kernel.termination()
        ##TODO: look at whether some cleaning functions needed for abides

    @abstractmethod
    def raw_state_to_state(self, raw_state: Dict[str, Any]) -> np.ndarray:
        """
        abstract method that transforms a raw state into a state representation

        Arguments:
            - raw_state: dictionnary that contains raw simulation information obtained from the gym experimental agent

        Returns:
            - state: state representation defining the MDP
        """
        raise NotImplementedError

    @abstractmethod
    def raw_state_to_reward(self, raw_state: Dict[str, Any]) -> float:
        """
        abstract method that transforms a raw state into the reward obtained during the step

        Arguments:
            - raw_state: dictionnary that contains raw simulation information obtained from the gym experimental agent

        Returns:
            - reward: immediate reward computed at each step
        """
        raise NotImplementedError

    @abstractmethod
    def raw_state_to_done(self, raw_state: Dict[str, Any]) -> float:
        """
        abstract method that transforms a raw state into the flag if an episode is done

        Arguments:
            - raw_state: dictionnary that contains raw simulation information obtained from the gym experimental agent

        Returns:
            - done: flag that describes if the episode is terminated or not
        """
        raise NotImplementedError

    @abstractmethod
    def raw_state_to_update_reward(self, raw_state: Dict[str, Any]) -> bool:
        """
        abstract method that transforms a raw state into the final step reward update (if needed)

        Arguments:
            - raw_state: dictionnary that contains raw simulation information obtained from the gym experimental agent

        Returns:
            - reward: update reward computed at the end of the episode
        """
        raise NotImplementedError

    @abstractmethod
    def raw_state_to_info(self, raw_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        abstract method that transforms a raw state into an info dictionnary

        Arguments:
            - raw_state: dictionnary that contains raw simulation information obtained from the gym experimental agent

        Returns:
            - reward: info dictionnary computed at each step
        """
        raise NotImplementedError
