from collections import namedtuple
from ipywidgets import widgets
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ray import tune
import torch
from tqdm import tqdm

from . import utils


Metric = namedtuple("metric", "p, key, transformation", defaults=(None,))

DEFAULT_METRICS = {
    "L2": Metric(2, "test/L2/current"),
    "L1": Metric(1, "test/L1/current"),
    "Linfinity": Metric(np.inf, "test/Linfinity/current"),
}

DEFAULT_KEYS = {
    # exp
    "trainable": "run",
    # config
    "wandb_mode": "wandb/mode",
    "wandb_api_key": "wandb/api_key",
    "seed": "seed",
    "target_fn": "target_fn",
    "n_samples": "n_samples",
    "plot_x": "plot_x",
    "plot_x_axis": "plot_x_axis",
    "dim": "dim",
    "log_path": "log_path",
    # tune analysis
    "iteration": "training_iteration",
    # file name
    "exp_file": "specs/exp_0.yaml",
}


def default_config2name(config, trial_df):
    """
    Transform a configuration into an algorithm name.
    """
    opt_name = utils.nested_get(config, "algorithm/optimizer_factory/__import__").split(
        "."
    )[-1]
    scheduler_name = (
        utils.nested_get(config, "algorithm/scheduler_factory/__import__")
        .split(".")[-1]
        .replace("LR", "")
    )

    model_type = utils.nested_get(config, "algorithm/model/__call__").split(".")[-1]
    if model_type == "MultilevelNet":
        model_name = "Multilevel"
    else:
        depth = utils.nested_get(config, "algorithm/model/depth")
        width = utils.nested_get(config, "algorithm/model/width")
        act = utils.nested_get(config, "algorithm/model/activation/__call__").split(
            "."
        )[-1]

        model_res = (
            "Res"
            if utils.nested_get(config, "algorithm/model/residual_connections", False)
            else ""
        )
        model_bn = (
            utils.nested_get(
                config, "algorithm/model/normalization_factory/__import__", ""
            )
            .split(".")[-1]
            .replace("None", "")
        )
        model_name = f"{depth}x{width}{model_res}{model_bn}{act}"

    bs = utils.nested_get(config, "algorithm/data_loader_kwargs/batch_size")
    epochs_per_iteration = utils.nested_get(config, "algorithm/epochs_per_iteration")
    lr = utils.nested_get(config, "algorithm/optimizer_kwargs/lr")
    params_file = utils.nested_get(config, "target_fn/params_file", None)
    if params_file:
        target_fn = Path(utils.nested_get(config, "target_fn/params_file")).stem
    else:
        target_fn = utils.nested_get(config, "target_fn/__call__").split(".")[-1]

    return {
        "algorithm": f"{model_name}_{opt_name}_{lr}{scheduler_name}_Bs{bs}x{epochs_per_iteration}",
        "target_fn": target_fn,
    }


class Visualizer:
    """
    Visualize experiments.
    """

    def __init__(
        self,
        exp_dirs,
        update_metrics=None,
        update_keys=None,
        save_path="plots",
        file_type="json",
        config2name=None,
    ):

        self.exp_dirs = [utils.absolute_path(d) for d in exp_dirs]
        self.analyses = [tune.ExperimentAnalysis(d) for d in self.exp_dirs]
        self.metrics = DEFAULT_METRICS
        if update_metrics is not None:
            self.metrics.update(update_metrics)
        self.keys = DEFAULT_KEYS
        if update_keys is not None:
            self.keys.update(update_keys)

        self.configs = {}
        self.trial_ckpt_paths = {}
        self.log_paths = {}
        self.exp_specs = {}
        self.trial_dataframes = {}
        self.config2name = config2name or default_config2name

        for analysis, exp_dir in zip(self.analyses, self.exp_dirs):
            analysis.set_filetype(file_type)
            configs = analysis.get_all_configs()
            self.configs.update(configs)

            trial_dataframes = {
                path: df
                for path, df in analysis.trial_dataframes.items()
                if not df.empty
            }
            self.trial_dataframes.update(trial_dataframes)

            self.trial_ckpt_paths.update(
                {
                    path: analysis.get_trial_checkpoints_paths(path)
                    for path in trial_dataframes
                }
            )
            config = list(configs.values())[0]
            log_path = utils.absolute_path(
                utils.nested_get(config, self.keys["log_path"])
            )
            exp_dir = str(exp_dir)
            self.log_paths[exp_dir] = log_path
            self.exp_specs[exp_dir] = utils.load_spec(log_path / self.keys["exp_file"])

        self.save_path = utils.absolute_path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        # defined in initialize
        self.df_all = None
        self.results = None
        self._initialized = False

    def _get_ckpt_path(self, path, iteration):
        ckpt_paths = list(
            filter(lambda x: x[1] == iteration, self.trial_ckpt_paths[path])
        )
        if len(ckpt_paths) >= 1:
            return ckpt_paths[0][0]

    @staticmethod
    def lower_bound(n_samples, dim, p, factor=1.0):
        return factor * (1 / n_samples) ** (1 / p + 1 / dim)

    def initialize(self):

        trial_dfs = []

        for path, trial_df in tqdm(
            self.trial_dataframes.items(), desc="Assemble trial dataframes"
        ):
            config = self.configs[path]
            # duplicates might occur due to resuming
            trial_df = trial_df.copy().drop_duplicates(
                keep="last",
                subset=self.keys["iteration"],
            )
            trial_df.loc[:, "n_samples"] = utils.nested_get(
                config, self.keys["n_samples"]
            )
            trial_df.loc[:, "seed"] = utils.nested_get(config, self.keys["seed"])
            trial_df.loc[:, "dim"] = utils.nested_get(config, self.keys["dim"])
            names = self.config2name(config, trial_df)
            trial_df.loc[:, "algorithm_name"] = names["algorithm"]
            trial_df.loc[:, "target_fn"] = names["target_fn"]
            trial_df.loc[:, "path"] = path
            trial_dfs.append(trial_df)

        df = pd.concat(trial_dfs, ignore_index=True)
        places = len(str(df.loc[:, self.keys["iteration"]].max()))
        df.loc[:, "algorithm"] = df.loc[
            :, ["algorithm_name", self.keys["iteration"]]
        ].apply(lambda x: f"{x[0]}_it{x[1]:0{places}d}", axis=1)
        df.loc[:, "checkpoint_path"] = df.loc[
            :, ["path", self.keys["iteration"]]
        ].apply(lambda x: self._get_ckpt_path(*x), axis=1)
        for name, metric in self.metrics.items():
            if metric.transformation is None:
                df.loc[:, name] = df.loc[:, metric.key]
            else:
                df.loc[:, name] = df.loc[:, metric.key].apply(metric.transformation)

        self.df_all = df.copy()

        # min_{algorithms} max_{targets} algorithms(targets)
        cols = ["target_fn", "n_samples", "algorithm"]
        df = df.loc[:, list(self.metrics.keys()) + cols]
        # mean over seeds
        df = (
            df.groupby(cols)
            .agg({name: lambda x: x.mean(skipna=False) for name in self.metrics})
            .reset_index()
        )
        # max over target_fns
        df_group = df.drop(columns="target_fn").groupby(["n_samples", "algorithm"])
        df_argmax = df_group.idxmax(skipna=False)

        # avg over target_fns
        df_avg = df_group.agg(
            {name: lambda x: x.mean(skipna=False) for name in self.metrics}
        ).reset_index()
        minavg_idx = df_avg.drop(columns=["algorithm"]).groupby("n_samples").idxmin()

        self.results = {}
        for name, metric in tqdm(
            self.metrics.items(), desc="Compute minmax dataframes"
        ):
            # minmax dataframe
            df_metric = df.loc[:, [name] + cols]

            df_max = df_metric.iloc[df_argmax.loc[:, name].dropna()].copy()

            minmax_idx = (
                df_max.drop(columns=["algorithm", "target_fn"])
                .groupby("n_samples")
                .idxmin()
                .loc[:, name]
                .dropna()
            )

            df_minmax = pd.merge(
                df_metric.iloc[minmax_idx],
                self.df_all,
                on=cols,
                suffixes=("_max", None),
            ).sort_values(by="n_samples")

            df_minmax.loc[:, "lower_bound"] = df_minmax.loc[
                :, ["n_samples", "dim"]
            ].apply(lambda x: self.lower_bound(*x, p=metric.p), axis=1)

            # minavg dataframe
            df_minavg = pd.merge(
                df_avg.iloc[minavg_idx.loc[:, name].dropna()].loc[
                    :, [name, "n_samples", "algorithm"]
                ],
                self.df_all,
                on=["n_samples", "algorithm"],
                suffixes=("_avg", None),
            ).sort_values(by="n_samples")

            # summary dataframe
            df_max.loc[:, "target_fn"] = "target_fns_max"
            df_summary = pd.concat([df_metric, df_max], ignore_index=True, join="inner")

            df_min = df_summary.groupby(
                ["n_samples", "target_fn"], as_index=False
            ).min()
            df_min.loc[:, "algorithm"] = "configs_min"
            df_summary = pd.concat(
                [df_summary, df_min], ignore_index=True, join="inner"
            )

            vmin = df_summary.loc[:, name].min()
            vmax = df_summary.loc[:, name].max()

            df_summary = df_summary.groupby("n_samples")
            summary = {
                n: df_summary.get_group(n)
                .pivot(index="algorithm", columns="target_fn", values=name)
                .style.background_gradient(axis=None, vmin=vmin, vmax=vmax)
                .format("{:2.1e}")
                for n in df_summary.groups
            }

            summary = {n: df for n, df in summary.items()}

            self.results[name] = {
                "minmax": df_minmax.copy(),
                "minavg": df_minavg.copy(),
                "summary": summary,
            }

        self._initialized = True

    def _restore_trainable(self, path, ckpt_path=None, disable_wandb=True):
        config = self.configs[path]
        if disable_wandb:
            utils.nested_update(
                config,
                {
                    self.keys["wandb_mode"]: "disabled",
                    self.keys["wandb_api_key"]: "demo",
                },
            )
        exp_dir = str(Path(path).parent)
        trainable_factory = utils.deserialize_spec(
            utils.nested_get(self.exp_specs[exp_dir], self.keys["trainable"])
        )
        trainable = trainable_factory(config)
        if ckpt_path is not None:
            trainable.load_checkpoint(ckpt_path)
        return trainable

    @staticmethod
    def _get_fit(x, y, x_eval=None):
        fit = np.polyfit(np.log(x), np.log(y), deg=1)
        x_eval = x_eval or np.linspace(min(x), max(x))
        y_fit = np.exp(fit[1]) * np.power(x_eval, fit[0])
        label = "$y={0:.3f}x^{{{1:.2f}}}$".format(np.exp(fit[1]), fit[0])
        return x_eval, y_fit, label

    def plot(
        self, metric_names=None, xtype="log", ytype="log", plot_x=None, plot_x_axis=None
    ):
        if not self._initialized:
            raise RuntimeError("Not initialized!")

        metric_names = metric_names or self.metrics.keys()
        fig = go.Figure()

        for i, metric_name in enumerate(metric_names):

            df_max = self.results[metric_name]["minmax"].drop_duplicates(
                subset="n_samples"
            )
            df_avg = self.results[metric_name]["minavg"].drop_duplicates(
                subset="n_samples"
            )

            for j, (df, k, dash) in enumerate(
                zip(
                    [df_avg, df_max],
                    [f"{metric_name}_avg", f"{metric_name}_max"],
                    ["dot", "solid"],
                )
            ):
                x = df.loc[:, "n_samples"]
                y = df.loc[:, k]
                x_fit, y_fit, label = Visualizer._get_fit(x, y)

                c = (2 * i + 1 + j) * 255 / (2 * len(metric_names) + 1)
                color = f"rgba({c},{255 - c},{c}, 0.8)"

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines+markers",
                        name=k,
                        line=dict(
                            color=color,
                            width=3,
                            dash=dash,
                        ),
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=x_fit,
                        y=y_fit,
                        mode="lines",
                        name=label,
                        line=dict(
                            color=color,
                            width=1,
                            dash="dot",
                        ),
                    )
                )

            fig.add_trace(
                go.Scatter(
                    x=df_max.loc[:, "n_samples"],
                    y=df_max.loc[:, "lower_bound"],
                    mode="lines",
                    line=dict(shape="linear", color=color, width=2, dash="dashdot"),
                    name=f"lower_bound_{metric_name}_max",
                )
            )

        n_algorithms = (
            self.df_all.loc[:, ["n_samples", "algorithm"]]
            .groupby("n_samples")
            .nunique()
            .min()
            .item()
        )
        n_targets = self.df_all.loc[:, "target_fn"].nunique()
        n_seeds = self.df_all.loc[:, "seed"].nunique()
        fig.update_layout(
            title=f"Minimum over >={n_algorithms} algorithms "
            f"(avg. over {n_seeds} seeds) of max./avg. error over {n_targets} targets.",
            xaxis_title="samples",
            yaxis_title="error",
            legend_title="metric",
        )
        fig.update_xaxes(type=xtype)
        fig.update_yaxes(type=ytype)
        fig.write_image(self.save_path / "minmax_summary.pdf")

        figs = [fig]
        for metric_name in metric_names:
            for row in self.results[metric_name]["minmax"].itertuples():
                if row.checkpoint_path is not None:
                    trainable = self._restore_trainable(row.path, row.checkpoint_path)
                    row_save_path = (
                        self.save_path / metric_name / f"samples={row.n_samples}"
                    )
                    row_save_path.mkdir(exist_ok=True, parents=True)
                    if plot_x is None:
                        config = self.configs[row.path]
                        plot_x = utils.deserialize_spec(
                            utils.nested_get(config, self.keys["plot_x"], default=None)
                        )
                        plot_x_axis = utils.deserialize_spec(
                            utils.nested_get(
                                config, self.keys["plot_x_axis"], default=None
                            )
                        )

                    fig = utils.visualize(
                        plot_x=plot_x,
                        plot_x_axis=plot_x_axis,
                        named_fns={
                            "target": trainable.target_fn,
                            "model": trainable.algorithm.model,
                        },
                        samples=trainable.algorithm.samples,
                        file=row_save_path / f"seed={row.seed}.pdf",
                        update_layout={
                            "title": f"{metric_name} | "
                            f"{trainable.algorithm.samples[0].shape[0]} samples | "
                            f"{row.algorithm_name} | "
                            f"seed {row.seed} | "
                            f"target_fn {row.target_fn}"
                        },
                    )
                    figs.append(fig)
        return figs

    def plot_target_fns(self, plot_x=None, plot_x_axis=None):
        if not self._initialized:
            raise RuntimeError("Not initialized!")

        fig = None
        target_fn_df = self.df_all.loc[:, ["path", "target_fn"]].drop_duplicates(
            subset="target_fn"
        )
        with torch.no_grad():
            for i, row in enumerate(target_fn_df.itertuples()):
                config = self.configs[row.path]
                if plot_x is None:
                    plot_x = utils.deserialize_spec(
                        utils.nested_get(config, self.keys["plot_x"], default=None)
                    )
                    plot_x_axis = utils.deserialize_spec(
                        utils.nested_get(config, self.keys["plot_x_axis"], default=None)
                    )

                target_fn = utils.deserialize_spec(
                    utils.nested_get(config, self.keys["target_fn"])
                )
                fig = utils.visualize(
                    plot_x=plot_x,
                    plot_x_axis=plot_x_axis,
                    named_fns={
                        row.target_fn: target_fn,
                    },
                    file=self.save_path / f"target_fns.pdf"
                    if i == len(target_fn_df) - 1
                    else None,
                    fig=fig,
                )
        return fig


def selector(path="results", name=None, sort=None):
    """
    Show all Tune results at a given path.
    """
    path = utils.absolute_path(path)
    dirs = [
        (d.name, d)
        for d in path.iterdir()
        if d.is_dir()
        and any(f.name.startswith("experiment_state") for f in d.iterdir())
        and (name is None or name in str(d))
    ]
    if sort == "name":
        dirs.sort(key=lambda d: d[0])
    elif sort == "time":
        dirs.sort(key=lambda d: d[1].stat().st_ctime, reverse=True)

    analyses_selection = widgets.SelectMultiple(
        options=dirs,
        description=f"Choose experiment from '{path.relative_to(utils.project_root())}':",
        style={"description_width": "initial"},
        layout={"width": "max-content"},
    )

    return analyses_selection
