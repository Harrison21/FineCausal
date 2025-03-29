import os
import yaml
import argparse
import shutil
from distutils.dir_util import copy_tree


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archs",
        type=str,
        choices=["FineParser"],
        default="FineParser",
        help="our approach",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["FineDiving"],
        default="FineDiving",
        help="dataset",
    )
    parser.add_argument("--prefix", type=str, default="default", help="experiment name")
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="resume training (interrupted by accident)",
    )
    parser.add_argument("--sync_bn", type=bool, default=False)
    parser.add_argument("--fix_bn", type=bool, default=True)
    parser.add_argument("--test", action="store_true", default=True)
    parser.add_argument("--ckpts", type=str, default="None", help="test used ckpt path")
    parser.add_argument(
        "--backup_code",
        action="store_true",
        default=True,
        help="backup code for reproducibility",
    )
    args = parser.parse_args()

    return args


def setup(args):

    args.config = "{}_FineParser.yaml".format(args.benchmark)
    args.experiment_path = os.path.join(
        "./experiments", args.archs, args.benchmark, args.prefix
    )
    if args.resume:
        cfg_path = os.path.join(args.experiment_path, "config.yaml")
        if not os.path.exists(cfg_path):
            print("Failed to resume")
            args.resume = False
            setup(args)
            return

        print("Resume yaml from %s" % cfg_path)
        with open(cfg_path) as f:
            config = yaml.load(f, Loader=yaml.Loader)
        merge_config(config, args)
        args.resume = True
    else:
        config = get_config(args)
        merge_config(config, args)
        create_experiment_dir(args)
        save_experiment_config(args)
        backup_code(args)


def get_config(args):
    try:
        print("Load config yaml from %s" % args.config)
        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.Loader)
    except:
        raise NotImplementedError("%s arch is not supported" % args.archs)
    return config


def merge_config(config, args):
    for k, v in config.items():
        setattr(args, k, v)


def create_experiment_dir(args):
    try:
        os.makedirs(args.experiment_path)
        print("Create experiment path successfully at %s" % args.experiment_path)
    except:
        pass


def save_experiment_config(args):
    config_path = os.path.join(args.experiment_path, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(args.__dict__, f)
        print("Save the Config file at %s" % config_path)


def backup_code(args):
    """
    Backup code files for reproducibility

    Args:
        args: arguments containing experiment path
    """
    code_backup_path = os.path.join(args.experiment_path, "code_backup")
    os.makedirs(code_backup_path, exist_ok=True)
    print("Backing up code to %s" % code_backup_path)

    # List of directories and files to backup
    backup_files = ["main.py", "launch.py"]
    backup_dirs = ["utils", "models", "tools", "datasets", "layers"]

    # Backup individual files
    for file_pattern in backup_files:
        for file in list_files_with_pattern(file_pattern):
            if os.path.exists(file):
                dst_file = os.path.join(code_backup_path, file)
                # Create subdirectory if needed
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                try:
                    shutil.copy2(file, dst_file)
                    print(f"- Backed up {file}")
                except Exception as e:
                    print(f"  Failed to backup {file}: {e}")

    # Backup directories
    for directory in backup_dirs:
        if os.path.exists(directory) and os.path.isdir(directory):
            dst_dir = os.path.join(code_backup_path, directory)
            try:
                copy_tree(directory, dst_dir)
                print(f"- Backed up directory {directory}")
            except Exception as e:
                print(f"  Failed to backup directory {directory}: {e}")

    print(f"Code backup completed at {code_backup_path}")


def list_files_with_pattern(pattern):
    """
    List files matching the pattern from the root directory

    Args:
        pattern: file pattern to match

    Returns:
        List of matched files
    """
    import glob

    return glob.glob(pattern)
