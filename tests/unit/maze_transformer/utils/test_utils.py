from maze_transformer.utils.utils import register_method


def test_register_method():
    class TestEvalsA:
        evals = {}

        @register_method(evals)
        @staticmethod
        def eval_function():
            pass

        @staticmethod
        def other_function():
            pass

    class TestEvalsB:
        evals = {}

        @register_method(evals)
        @staticmethod
        def other_eval_function():
            pass

    evalsA = TestEvalsA.evals
    evalsB = TestEvalsB.evals
    assert list(evalsA.keys()) == ["eval_function"]
    assert list(evalsB.keys()) == ["other_eval_function"]
