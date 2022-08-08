import os
import subprocess
import shutil
from jill.utils.interactive_utils import query_yes_no
from jill.install import install_julia

def get_jnumpy_dir():
    return os.getenv("JNUMPY_HOME", os.path.expanduser("~/.jnumpy"))

def get_symlink_dir():
    return os.path.join(get_jnumpy_dir(), "bin")

def get_julia_exe():
    julia_exepath = os.getenv("TYPY_JL_EXE")
    if not julia_exepath:
        julia_exepath = get_default_julia_exe()
    return julia_exepath

def get_default_julia_exe():
    julia_exepath = shutil.which("julia")
    ver_cmd = [julia_exepath, "--version"]
    try:
        subprocess.check_output(ver_cmd)
    except:
        quest = "Can not find julia.\nWould you like jnumpy to install julia now?"
        to_continue = query_yes_no(quest)
        if not to_continue:
            raise Exception("can't find julia.")
        else:
            setup_julia()
            serach_path = get_symlink_dir()
            print(serach_path)
            julia_exepath = shutil.which("julia", path=serach_path)
    return julia_exepath

def get_project_args():
    default_envs_dir = os.path.join(get_jnumpy_dir(), "envs/default")
    return f"--project={default_envs_dir}"

def setup_julia(version=None):
    print("installing julia with jill\n")
    install_dir = os.path.join(get_jnumpy_dir(), "julias")
    symlink_dir = get_symlink_dir()
    install_julia(version=version, install_dir=install_dir, symlink_dir=symlink_dir, confirm=True)
