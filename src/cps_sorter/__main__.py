"""Allow cookiecutter to be executable through `python -m cookiecutter`."""
from cps_sorter.cli import main


if __name__ == "__main__":  # pragma: no cover
    main(prog_name="cps_sorter")