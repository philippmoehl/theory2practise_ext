from argparse import ArgumentParser

from theory2practice.utils import load_spec, setup


def main(argv=None):
    """Parse arguments and run the given experiments."""

    parser = ArgumentParser(description="Tune executor")
    parser.add_argument(
        "-e",
        "--exp-files",
        nargs="+",
        type=str,
        default=["specs/pde_reaction/exp_0.yaml"],
        help="Experiment spec files.",
    )
    parser.add_argument(
        "-r",
        "--runner-file",
        type=str,
        default="specs/runner.yaml",
        help="Experiment runner spec file.",
    )

    args = parser.parse_args(argv)
    setup()
    runner = load_spec(args.runner_file, deserialize=True)
    runner.run(args.exp_files)


if __name__ == "__main__":
    main()
