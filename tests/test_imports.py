def test_public_imports():
    import memmark
    from memmark.backends import load_amem, load_graphiti
    from memmark.benchmarks.locomo import load_locomo
    from memmark.examples.run_locomo_full import build_parser

    assert memmark.MemoryWatermarker is not None
    assert load_amem is not None
    assert load_graphiti is not None
    assert load_locomo is not None
    assert build_parser().parse_args(["--backend", "amem"]).backend == "amem"
    assert build_parser().parse_args(["--backend", "graphiti"]).backend == "graphiti"
