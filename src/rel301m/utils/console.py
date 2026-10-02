"""Terminal progress and narrowly scoped warnings for the frozen Panda/BASIC stack."""

import logging
import math
import shutil
import sys
import time


class PandaWarningFilter(logging.Filter):
    """Retain unknown warnings and every error; skip only known unused features."""

    def filter(self, record):
        if record.name != 'robosuite_logs' or record.levelno != logging.WARNING:
            return True
        message = record.getMessage()
        if record.filename == 'macros.py':
            return not (message in ('No private macro file found!',
                                   'It is recommended to use a private macro file') or
                        (message.startswith('To setup, run: python ') and
                         message.endswith('/scripts/setup_macros.py')))
        if record.filename == '__init__.py':
            return not (message.startswith('Could not import robosuite_models. Some robots may not be available.') or
                        message.startswith('Could not load the mink-based whole-body IK.') and
                        'GR1 robot' in message)
        if record.filename == 'robot.py':
            for part in ('left', 'torso', 'head', 'base', 'legs'):
                known = (f'The config has defined for the controller "{part}", '
                         'but the robot does not have this component. Skipping, but make sure this is intended.'
                         f'Removing the controller config for {part} from self.part_controller_config.')
                if message == known:
                    return False
        return True


def install_panda_warning_filter():
    logger = logging.getLogger('robosuite_logs')
    if not any(isinstance(item, PandaWarningFilter) for item in logger.filters):
        logger.addFilter(PandaWarningFilter())


def duration(seconds):
    if seconds is None:
        return '?'
    hours, remainder = divmod(max(0, math.ceil(seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


class TrainingProgress:
    """Redraw in a terminal; write readable periodic lines through pipes / tee."""

    def __init__(self, total, seed, *, enabled=True, stream=None, clock=None):
        self.total, self.seed, self.enabled = total, seed, enabled
        self.stream = stream if stream is not None else sys.stderr
        self.clock = clock if clock is not None else time.monotonic
        self.interactive = self.stream.isatty()
        self.start = self.clock()
        self.last_render = -float('inf')
        self.completed, self.stage, self.detail = 0, 'initializing', ''
        self.eval_count, self.eval_total = 0, None
        self.closed = False
        self.line_width = 0

    def set_phase(self, stage, *, episodes=None):
        self.stage, self.eval_count, self.eval_total = stage, 0, episodes
        self.render(force=True)

    def episode_completed(self, count, total):
        self.eval_count, self.eval_total = count, total
        self.render(force=count == total)

    def update(self, completed, detail=''):
        self.completed, self.detail = completed, detail
        self.render()

    def render(self, *, force=False):
        if not self.enabled or self.closed:
            return
        now = self.clock()
        if not force and now - self.last_render < (1 if self.interactive else 10):
            return
        elapsed = now - self.start
        rate = self.completed / elapsed if elapsed > 0 else 0
        eta = (self.total - self.completed) / rate if rate > 0 else None
        filled = min(20, int(20 * self.completed / self.total))
        bar = '#' * filled + '-' * (20 - filled)
        phase = self.stage + (f' {self.eval_count}/{self.eval_total}' if self.eval_total else '')
        eta_label = 'finalizing' if self.completed == self.total and self.stage not in ('completed', 'FAILED', 'stopped') else duration(eta)
        line = (f'seed={self.seed} [{bar}] {self.completed / self.total:6.2%} '
                f'{self.completed:,}/{self.total:,} | {rate:.1f} steps/s | '
                f'elapsed {duration(elapsed)} | ETA {eta_label} | {phase} {self.detail}').rstrip()
        if self.interactive and len(line) >= shutil.get_terminal_size().columns:
            short_phase = {'initializing': 'init', 'random baseline': 'random',
                           'evaluation': 'eval', 'training': 'train'}.get(self.stage, self.stage)
            if self.eval_total:
                short_phase += f' {self.eval_count}/{self.eval_total}'
            compact_bar = '#' * (filled // 2) + '-' * (10 - filled // 2)
            line = (f'seed={self.seed} [{compact_bar}] {self.completed / self.total:.0%} '
                    f'{self.completed:,}/{self.total:,} {rate:.0f}/s ETA {eta_label} | {short_phase}')
        if self.interactive:
            self.stream.write('\r' + line.ljust(self.line_width))
            self.line_width = len(line)
        else:
            self.stream.write(line + '\n')
        self.stream.flush()
        self.last_render = now

    def write(self, message):
        if self.enabled and self.interactive and self.line_width:
            self.stream.write('\r' + ' ' * self.line_width + '\r')
            self.stream.flush()
        print(message, flush=True)
        self.render(force=True)

    def finish(self, status):
        self.stage, self.eval_total = status, None
        self.render(force=True)
        if self.enabled and self.interactive:
            self.stream.write('\n')
            self.stream.flush()
        self.closed = True

    def close(self):
        if not self.closed:
            self.finish('stopped')
