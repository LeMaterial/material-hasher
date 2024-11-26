from typing import Optional


def get_all_test_cases() -> list[str]:
    """Utility function to get a list of all available test cases.

    Returns
    -------
    list[str]
        List of all available test cases.
    """
    return ["gaussian_noise"]


def make_test_cases(
    test_cases: Optional[list[str]] = None,
    ignore_test_cases: Optional[list[str]] = None,
) -> list[str]:
    """Utility function to generate a list of test cases to run based on the specified test cases and ignored test cases.

    The procedure is as follows:

    1. If ``test_cases`` is not ``None``, include only the specified test cases.
    2. Otherwise, if ``test_cases`` is ``None``, include all test cases (from :func:`get_all_test_cases`).
    3. If ``ignore_test_cases`` is not ``None``, filter the list of test cases to exclude the specified test cases.

    Parameters
    ----------
    test_cases : Optional[list[str]], optional
        List of test cases the user wants, by default ``None``
    ignore_test_cases : Optional[list[str]], optional
        List of test to ignore, by default ``None``

    Returns
    -------
    list[str]
        List of test cases to run.

    Raises
    ------
    ValueError
        If an unknown test case is specified in ``test_cases`` or ``ignore_test_cases``.
    ValueError
        If the resulting list of test cases is empty.
    """
    all_test_cases = get_all_test_cases()

    if test_cases is None:
        test_cases = all_test_cases

    if ignore_test_cases is None:
        ignore_test_cases = []

    for t in test_cases + ignore_test_cases:
        if t not in all_test_cases:
            raise ValueError(f"Unknown test case: {t}")

    if test_cases is not None:
        all_test_cases = [tc for tc in all_test_cases if tc in test_cases]
    if ignore_test_cases is not None:
        all_test_cases = [tc for tc in all_test_cases if tc not in ignore_test_cases]

    if not all_test_cases:
        raise ValueError("No test cases to run.")

    return all_test_cases


def get_test_data(test_case: str) -> dict:
    """Utility function to get test data for a given test case.

    Parameters
    ----------
    test_case : str
        Name of the test case.

    Returns
    -------
    dict
        Dictionary of test data.
    """
    if test_case == "gaussian_noise":
        return {"data": "gaussian noise data"}
    else:
        raise ValueError(f"Unknown test case: {test_case}")
