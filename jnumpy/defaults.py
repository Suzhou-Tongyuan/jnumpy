from __future__ import annotations
import os
import subprocess
import shutil
import jnumpy.envars as envars
import warnings


def get_jnumpy_dir():
    return os.getenv(envars.CF_JNUMPY_HOME, os.path.expanduser("~/.jnumpy"))


def get_symlink_dir():
    """Get the directory where julia is symlinked to."""
    return os.path.join(get_jnumpy_dir(), "bin")


def setup_julia_exe_():
    try:
        envars.SessionCtx.JULIA_EXE
        return
    except AttributeError:
        pass
    julia_exepath = os.getenv("TYPY_JL_EXE")
    if not julia_exepath:
        julia_exepath = get_default_julia_exe()
    envars.SessionCtx.JULIA_EXE = julia_exepath
    return


def check_valid_julia_exe(julia_exepath):
    if julia_exepath:
        try:
            ver_cmd = [julia_exepath, "--version"]
            subprocess.check_output(ver_cmd)
        except subprocess.CalledProcessError:
            return None
    return julia_exepath


def get_default_julia_exe() -> str:
    # search julia in sys path
    julia_exepath = shutil.which("julia")
    julia_exepath = check_valid_julia_exe(julia_exepath)
    serach_path = get_symlink_dir()
    if not julia_exepath:
        # search julia in default jnumpy path
        julia_exepath = shutil.which("julia", path=serach_path)
        julia_exepath = check_valid_julia_exe(julia_exepath)

    if not julia_exepath:
        assure_jill()
        from jill.utils.interactive_utils import query_yes_no

        quest = "Can not find julia.\nWould you like jnumpy to install julia now?"
        to_continue = query_yes_no(quest)
        if not to_continue:
            raise EnvironmentError("cannot find julia.")
        else:
            setup_julia()
            julia_exepath = shutil.which("julia", path=serach_path)
            if not julia_exepath:
                raise EnvironmentError(
                    "Julia has been installed but cannot get found! Please issue a bug report!"
                )
    return julia_exepath


def get_project_args():
    default_envs_dir = envars.DefaultProj_directory
    return f"--project={default_envs_dir}"


def setup_julia(version=None):
    print("installing julia with jill...\n")
    install_dir = os.path.join(get_jnumpy_dir(), "julias")
    symlink_dir = get_symlink_dir()

    assure_jill()
    from jill.install import install_julia

    install_julia(
        version=version, install_dir=install_dir, symlink_dir=symlink_dir, confirm=False
    )


def assure_jill():
    try:
        import jill
    except ImportError:
        warnings.warn(
            "jill is not installed, but `jill` (auto-install julia tool) is missing!\n"
        )
        raise RuntimeError(
            "Julia is not available at this machine, but automatic Julia installation is not available due to:\n"
            "    jill is missing.\n"
            "Possible fix: `pip install jill`."
        )
