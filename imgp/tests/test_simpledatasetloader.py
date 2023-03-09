from datasets.simpledatasetloader import SimpleDatasetLoader


def test_constructor():
    a = SimpleDatasetLoader(preprocessors=["a", "b", "c"])
    assert a.preprocessors == ["a", "b", "c"]

    b = SimpleDatasetLoader(preprocessors=None)
    assert b.preprocessors == []

    c = SimpleDatasetLoader()
    assert c.preprocessors == []
