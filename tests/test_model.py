from trainer import model


def test_build_model():
    """

    """
    m = model.build_model()
    assert m is not None
    assert m.layers is not None
    assert m.layers != []