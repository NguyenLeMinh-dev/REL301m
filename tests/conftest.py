import pytest

from rel301m.envs.robosuite_factory import make_two_arm_lift


@pytest.fixture(scope="session")
def env():
    instance = make_two_arm_lift()
    try:
        yield instance
    finally:
        instance.close()
