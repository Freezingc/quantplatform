"""Package-level contract tests."""


def test_package_exposes_a_non_empty_version() -> None:
    import research_v2

    assert isinstance(research_v2.__version__, str)
    assert research_v2.__version__
