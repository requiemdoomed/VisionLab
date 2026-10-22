"""CPU dependency choices for supported desktop platforms."""
import platform
import sys


def model_install_command(system=None, machine=None):
    system = system or platform.system()
    machine = (machine or platform.machine()).lower()
    if system not in {'Windows', 'Darwin', 'Linux'}:
        raise ValueError(f'Unsupported operating system: {system}.')
    if machine not in {'x86_64', 'amd64', 'arm64', 'aarch64'}:
        raise ValueError('Use 64-bit Python.')
    if system == 'Windows' and machine not in {'x86_64', 'amd64'}:
        raise ValueError('Use Windows x64. Windows ARM is not verified.')
    intel_mac = system == 'Darwin' and machine in {'x86_64', 'amd64'}
    torch_version, vision_version = ('2.2.2', '0.17.2') if intel_mac else ('2.7.0', '0.22.0')
    command = [sys.executable, '-m', 'pip', 'install', f'torch=={torch_version}', f'torchvision=={vision_version}']
    if system != 'Darwin':
        command += ['--index-url', 'https://download.pytorch.org/whl/cpu']
    return command
