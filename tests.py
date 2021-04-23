import pytest
import datetime
import git
import sys
import argparse
from pathlib import Path


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t",
        "--text_file_output",
        action="store_true",
        help="Should the test output be saved as a text file (rather than the default junit xml)?",
        required=False,
    )

    args = parser.parse_args()

    output_folder = Path("tests/outputs/")

    if not output_folder.exists():

        output_folder.mkdir()

    if args.text_file_output:

        repo = git.Repo(search_parent_directories=True)

        sha = repo.head.object.hexsha

        log_file = output_folder.joinpath(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f") + "__tests.txt"
        )

        sys.stdout = open(log_file, "w")

        pytest.main(args=["tests"])

        print("\ncommit sha: " + sha)

    else:

        log_file = output_folder.joinpath(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f") + "__tests.xml"
        )

        pytest.main(args=["tests", "--junitxml", str(log_file)])
