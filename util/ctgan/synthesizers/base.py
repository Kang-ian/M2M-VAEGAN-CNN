"""BaseSynthesizer module."""

import contextlib

import numpy as np
import torch


@contextlib.contextmanager
#上下文管理器
# 功能:管理随机状态的上下文管理器。
# 在进入上下文时，设置 NumPy 和 PyTorch 的随机状态为指定的状态；在退出上下文时，恢复原始的随机状态，并更新模型的随机状态。
def set_random_states(random_state, set_model_random_state):
    """Context manager for managing the random state.

    Args:
        random_state (int or tuple):
            The random seed or a tuple of (numpy.random.RandomState, torch.Generator).
        set_model_random_state (function):
            Function to set the random state on the model.
    """
    original_np_state = np.random.get_state()
    original_torch_state = torch.get_rng_state()

    random_np_state, random_torch_state = random_state

    np.random.set_state(random_np_state.get_state())
    torch.set_rng_state(random_torch_state.get_state())

    try:
        yield
    finally:
        current_np_state = np.random.RandomState()
        current_np_state.set_state(np.random.get_state())
        current_torch_state = torch.Generator()
        current_torch_state.set_state(torch.get_rng_state())
        set_model_random_state((current_np_state, current_torch_state))

        np.random.set_state(original_np_state)
        torch.set_rng_state(original_torch_state)

#一个装饰器
# 用于在调用被装饰的函数之前设置随机状态。如果 self.random_states 不为 None，则使用 set_random_states 上下文管理器设置随机状态。
def random_state(function):
    """Set the random state before calling the function.

    Args:
        function (Callable):
            The function to wrap around.
    """

    def wrapper(self, *args, **kwargs):
        if self.random_states is None:
            return function(self, *args, **kwargs)

        else:
            with set_random_states(self.random_states, self.set_random_state):
                return function(self, *args, **kwargs)

    return wrapper


class BaseSynthesizer:
    """Base class for all default synthesizers of ``CTGAN``."""

    random_states = None#初始随机状态
    #改进 BaseSynthesizer 的序列化过程。在序列化之前，将模型移动到 CPU 上，并将随机状态保存为字典。
    def __getstate__(self):
        """Improve pickling state for ``BaseSynthesizer``.

        Convert to ``cpu`` device before starting the pickling process in order to be able to
        load the model even when used from an external tool such as ``SDV``. Also, if
        ``random_states`` are set, store their states as dictionaries rather than generators.

        Returns:
            dict:
                Python dict representing the object.
        """
        device_backup = self._device
        self.set_device(torch.device('cpu'))
        state = self.__dict__.copy()
        self.set_device(device_backup)
        if (
            isinstance(self.random_states, tuple)
            and isinstance(self.random_states[0], np.random.RandomState)
            and isinstance(self.random_states[1], torch.Generator)
        ):
            state['_numpy_random_state'] = self.random_states[0].get_state()
            state['_torch_random_state'] = self.random_states[1].get_state()
            state.pop('random_states')

        return state
    #恢复 BaseSynthesizer 的状态。从状态字典中恢复随机状态，并根据当前硬件设置设备。
    def __setstate__(self, state):
        """Restore the state of a ``BaseSynthesizer``.

        Restore the ``random_states`` from the state dict if those are present and then
        set the device according to the current hardware.
        """
        if '_numpy_random_state' in state and '_torch_random_state' in state:
            np_state = state.pop('_numpy_random_state')
            torch_state = state.pop('_torch_random_state')

            current_torch_state = torch.Generator()
            current_torch_state.set_state(torch_state)

            current_numpy_state = np.random.RandomState()
            current_numpy_state.set_state(np_state)
            state['random_states'] = (current_numpy_state, current_torch_state)

        self.__dict__ = state
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.set_device(device)
    #将模型保存到指定的路径。在保存之前，将模型移动到 CPU 上，以避免设备不兼容问题。
    def save(self, path):
        """Save the model in the passed `path`."""
        # 确保路径包含文件名和.pth扩展名
        if not path.endswith('.pth'):
            path += '.pth'
        device_backup = self._device# 保存当前模型所在的设备（CPU或GPU）到变量device_backup中，以便之后可以恢复。
        self.set_device(torch.device('cpu'))# 将模型移动到CPU上，这样做通常是为了避免在保存模型时出现设备不兼容的问题。
        torch.save(self, path)#使用PyTorch的save函数将模型保存到指定的路径path。
        self.set_device(device_backup)#将模型恢复到之前保存的设备上。

    @classmethod
    def load(cls, path):#" 是load方法的文档字符串，说明这个方法的作用是从一个指定的路径加载模型。
        """Load the model stored in the passed `path`."""
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')#确定加载模型的设备，如果有可用的GPU，则使用GPU，否则使用CPU。
        model = torch.load(path)#使用PyTorch的load函数从指定的路径path加载模型。
        model.set_device(device)# 将加载的模型移动到之前确定的设备上。
        return model
    #设置随机状态。可以接受整数作为随机种子，或者直接接受一个包含 numpy.random.RandomState 和 torch.Generator 的元组。
    def set_random_state(self, random_state):
        """Set the random state.

        Args:
            random_state (int, tuple, or None):
                Either a tuple containing the (numpy.random.RandomState, torch.Generator)
                or an int representing the random seed to use for both random states.
        """
        if random_state is None:
            self.random_states = random_state
        elif isinstance(random_state, int):
            self.random_states = (
                np.random.RandomState(seed=random_state),
                torch.Generator().manual_seed(random_state),
            )
        elif (
            isinstance(random_state, tuple)
            and isinstance(random_state[0], np.random.RandomState)
            and isinstance(random_state[1], torch.Generator)
        ):
            self.random_states = random_state
        else:
            raise TypeError(
                f'`random_state` {random_state} expected to be an int or a tuple of '
                '(`np.random.RandomState`, `torch.Generator`)'
            )

