# CONTRIBUTING

----

Thanks for your interest in contributing to this package! We're hoping it can be made even better through community contributions.

## Requests and feedback

For any bugs, issues or feature requests please open an [issue](https://github.com/lvgig/tubular/issues) on the project.

## Requirements for contributions

We have some general requirements for all contributions then specific requirements when adding completely new transformers to the package. This is to ensure consistency with the existing codebase.

### Code formatting

Contributions should be formatted with `black`. There is a `pre-commit` file in the `.githooks` folder that can be activated for your local clone of the repo with `git config core.hooksPath .githooks`.

### Tests

All existing tests must pass and new functionality added must be tested. Tests must be written with [pytest](https://docs.pytest.org/en/stable/). The tests for existing transformers give great examples to work from that show what is expected to be covered in the tests.

### Docstrings

Docstrings need to be updated or added for new functionality.

### New transformers

To be consistent with `scikit-learn` - all transformers will implement at least an `__init__` and `transform(X)` method, then a `fit(X, y=None)` method if there is something to learn from the training data and potentially an `reverse_transform(X)` method too. We can group the common functionality of transformers we expect to be tested together by method;

## List of contributors

- [munichpavel](https://github.com/munichpavel)
- [bissoligiulia](https://github.com/bissoligiulia)
- [ClaireF57](https://github.com/ClaireF57)

Prior to the open source release of the package there have been contributions from many individuals in the LV GI Data Science team;

- Richard Angell
- Ned Webster
- Dapeng Wang
- David Silverstone
- Shreena Patel
- Angelos Charitidis
- David Hopkinson
- Liam Holmes
- Sandeep Karkhanis
- KarHor Yap
- Alistair Rogers
- Maria Navarro
- Marek Allen
- James Payne
