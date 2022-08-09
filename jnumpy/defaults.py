import os
import subprocess
import shutil
import jnumpy.envars as envars
from jill.utils.interactive_utils import query_yes_no
from jill.install import install_julia


def get_jnumpy_dir():
    return os.getenv(envars.CF_JNUMPY_HOME, os.path.expanduser("~/.jnumpy"))


def get_symlink_dir():
    """Get the directory where julia is symlinked to."""
    return os.path.join(get_jnumpy_dir(), "bin")


def get_julia_exe():
    julia_exepath = os.getenv("TYPY_JL_EXE")
    if not julia_exepath:
        julia_exepath = get_default_julia_exe()
    return julia_exepath


def get_default_julia_exe() -> str:
    julia_exepath = shutil.which("julia")
    if julia_exepath:
        try:
            ver_cmd = [julia_exepath, "--version"]
            subprocess.check_output(ver_cmd)
        except subprocess.CalledProcessError:
            julia_exepath = None

    if not julia_exepath:
        quest = "Can not find julia.\nWould you like jnumpy to install julia now?"
        to_continue = query_yes_no(quest)
        if not to_continue:
            raise EnvironmentError("cannot find julia.")
        else:
            setup_julia()
            serach_path = get_symlink_dir()
            julia_exepath = shutil.which("julia", path=serach_path)
            if not julia_exepath:
                raise EnvironmentError(
                    "Julia has been installed but cannot get found! Please issue a bug report!"
                )
    return julia_exepath


def get_project_args():
    default_envs_dir = os.path.join(get_jnumpy_dir(), "envs/default")
    return f"--project={default_envs_dir}"


def setup_julia(version=None):
    print("installing julia with jill...\n")
    install_dir = os.path.join(get_jnumpy_dir(), "julias")
    symlink_dir = get_symlink_dir()
    install_julia(
        version=version, install_dir=install_dir, symlink_dir=symlink_dir, confirm=False
    )
