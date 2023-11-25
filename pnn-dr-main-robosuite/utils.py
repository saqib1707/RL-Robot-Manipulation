# -*- coding: utf-8 -*-
import os
import plotly
from plotly.graph_objs import Scatter, Line
import numpy as np
import torch
from torch import multiprocessing as mp

# Global counter
class Counter:
    def __init__(self):
        """
        Class constructor.
        """
        self.val = mp.Value("i", 0)
        self.lock = mp.Lock()

    def increment(self):
        """
        Increments in one unit the counter value.
        """
        with self.lock:
            self.val.value += 1

    def value(self):
        """
        Obtain the counter value.

        Returns:
            int: counter value.
        """
        with self.lock:
            return self.val.value


def state_to_tensor(state, device):
    """
    Converts a state from the OpenAI Gym (a numpy array) to a batch tensor.

    Args:
        state (tuple): joints' variables and image observation.

    Returns:
        tuple: non-rgb state and rgb state tensors.
    """
    # Copies numpy arrays as they have negative strides
    # return (
    #     torch.Tensor(state[0]).unsqueeze(0),
    #     (torch.from_numpy(state[1].copy())).permute(2, 1, 0).float().div_(255).unsqueeze(0),
    # )

    return torch.from_numpy(state.copy()).permute(2, 1, 0).float().div_(255).unsqueeze(0).to(device)

    # print("Before:", state.shape, state.dtype, state.min(), state.max())
    # tmp1 = torch.from_numpy(state.copy())
    # print("Tmp1:", tmp1.shape, tmp1.dtype, tmp1.min(), tmp1.max())
    # tmp2 = tmp1.permute(2,1,0)
    # print("Tmp2:", tmp2.shape, tmp2.dtype, tmp2.min(), tmp2.max())
    # tmp3 = tmp2.float()
    # print("Tmp3:", tmp3.shape, tmp3.dtype, tmp3.min(), tmp3.max())
    # tmp4 = tmp3.div_(255)
    # print("Tmp4:", tmp4.shape, tmp4.dtype, tmp4.min(), tmp4.max())
    # tmp5 = tmp4.unsqueeze(0)
    # print("Tmp5:", tmp5.shape, tmp5.dtype, tmp5.min(), tmp5.max())
    # return tmp5


def plot_line(xs, ys_population):
    """
    Plots min, max and mean + standard deviation bars of a population over time.

    Args:
        xs (list): values for the x-axis.
        ys_population (list): values for the y-axis.
    """
    max_colour = "rgb(0, 132, 180)"
    mean_colour = "rgb(0, 172, 237)"
    std_colour = "rgba(29, 202, 255, 0.2)"

    ys = torch.Tensor(ys_population)
    ys_min = ys.min(1)[0].squeeze()
    ys_max = ys.max(1)[0].squeeze()
    ys_mean = ys.mean(1).squeeze()
    ys_std = ys.std(1).squeeze()
    ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

    trace_max = Scatter(x=xs, y=ys_max.numpy(), line=Line(color=max_colour, dash="dash"), name="Max")
    trace_upper = Scatter(
        x=xs,
        y=ys_upper.numpy(),
        line=Line(color="transparent"),
        name="+1 Std. Dev.",
        showlegend=False,  # transparent
    )
    trace_mean = Scatter(
        x=xs,
        y=ys_mean.numpy(),
        fill="tonexty",
        fillcolor=std_colour,
        line=Line(color=mean_colour),
        name="Mean",
    )
    trace_lower = Scatter(
        x=xs,
        y=ys_lower.numpy(),
        fill="tonexty",
        fillcolor=std_colour,
        line=Line(color="transparent"),  # transparent
        name="-1 Std. Dev.",
        showlegend=False,
    )
    trace_min = Scatter(x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash="dash"), name="Min")

    plotly.offline.plot(
        {
            "data": [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
            "layout": dict(
                title="Rewards",
                xaxis={"title": "Step"},
                yaxis={"title": "Average Reward"},
            ),
        },
        filename=os.path.join("results", "rewards.html"),
        auto_open=False,
    )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)