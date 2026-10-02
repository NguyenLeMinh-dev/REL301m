import io
import logging

import pytest

from rel301m.utils.console import PandaWarningFilter, TrainingProgress


def record(filename, message, level=logging.WARNING):
    return logging.LogRecord('robosuite_logs', level, '/simulator/' + filename, 1, message, (), None)


@pytest.mark.parametrize('filename,message', [
    ('macros.py', 'No private macro file found!'),
    ('macros.py', 'It is recommended to use a private macro file'),
    ('macros.py', 'To setup, run: python /venv/robosuite/scripts/setup_macros.py'),
    ('__init__.py', 'Could not import robosuite_models. Some robots may not be available. Optional models missing.'),
    ('__init__.py', 'Could not load the mink-based whole-body IK. Missing default setting for GR1 robot.'),
    ('robot.py', 'The config has defined for the controller "left", but the robot does not have this component. '
     'Skipping, but make sure this is intended.Removing the controller config for left from self.part_controller_config.'),
])
def test_only_known_optional_notices_are_filtered(filename, message):
    quiet = PandaWarningFilter()
    assert not quiet.filter(record(filename, message))
    assert quiet.filter(record(filename, message, logging.ERROR))
    assert quiet.filter(record('other_module.py', message))


def test_unknown_and_controller_numeric_warnings_remain_visible():
    quiet = PandaWarningFilter()
    for message in ('MuJoCo: Nan or Inf in QACC', 'Singular mass matrix',
                    'Unexpected controller saturation', 'Unknown optional-feature failure'):
        assert quiet.filter(record('robot.py', message))
    assert quiet.filter(record('__init__.py', 'Could not load the mink-based whole-body IK. Unexpected Panda problem.'))


def test_progress_has_rate_eta_and_failure_does_not_claim_completion():
    stream = io.StringIO()
    now = [0.]
    progress = TrainingProgress(100, 2, stream=stream, clock=lambda: now[0])
    now[0] = 10.
    progress.update(25, 'updates=12')
    progress.finish('FAILED')
    output = stream.getvalue()
    assert '25.00%' in output and '25/100' in output
    assert '2.5 steps/s' in output and 'ETA 00:00:30' in output
    assert 'FAILED' in output and 'completed' not in output and '100.00%' not in output


def test_progress_final_evaluation_is_visible_before_completed():
    stream = io.StringIO()
    now = [0.]
    progress = TrainingProgress(100, 0, stream=stream, clock=lambda: now[0])
    now[0] = 10.
    progress.update(100)
    progress.set_phase('evaluation', episodes=50)
    progress.episode_completed(50, 50)
    before = stream.getvalue()
    assert 'evaluation 50/50' in before and 'ETA finalizing' in before
    assert 'completed' not in before
    progress.finish('completed')
    assert stream.getvalue().splitlines()[-1].endswith('completed')


def test_progress_can_be_disabled():
    stream = io.StringIO()
    progress = TrainingProgress(100, 0, stream=stream, enabled=False)
    progress.set_phase('training')
    progress.update(100)
    progress.finish('completed')
    assert stream.getvalue() == ''
