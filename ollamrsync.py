import json
from json import JSONEncoder
import os
import argparse
import platform
from pathlib import Path
from contextlib import contextmanager
import sys
from urllib.parse import urlparse
import requests
import subprocess
import signal
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper

@contextmanager
def optional_dependencies(error: str = "ignore"):
    assert error in {"raise", "warn", "ignore"}
    try:
        yield None
    except ImportError as e:
        if error == "raise":
            msg = f'Missing required dependency "{e.name}". Use pip or conda to install.'
            print(f'Error: {msg}')
            raise e
        if error == "warn":
            msg = f'Missing optional dependency "{e.name}". Use pip or conda to install.'
            print(f'Warning: {msg}')
        if error == "ignore":
            pass

parser = argparse.ArgumentParser(prog='ollamarsync', description="Copy local Ollama models to a remote instance", epilog='Text at the bottom of help')
parser.add_argument('local_model', type=str,
                    help='Source local model to copy eg. mistral:latest')
parser.add_argument('remote_server', type=str,
                    help='Remote ollama server eg. http://192.168.0.100:11434')

args = parser.parse_args()

thisos = platform.system()

def get_env_var(var_name, default_value):
    return os.environ.get(var_name, default_value)

def get_platform_path(input_path):
    if input_path != "*":
        return input_path
    else:
        if thisos == "Windows":
            return f'{os.environ["USERPROFILE"]}{separator}.ollama{separator}models'
        elif thisos == "Darwin":
            return "~/.ollama/models"
        else:
            return "/usr/share/ollama/.ollama/models"

def get_platform_separator():
    if thisos == "Windows":
        return "\\"
    return "/"

def get_digest_separator():
    return "-"

def model_base(model_name):
    parts = model_name.split('/', 1)
    if "/" in model_name:
        return parts[0]
    else:
        return ""

def validate_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme in ['http', 'https'], result.port, not result.query, not result.path, not result.path.endswith('/')])
    except ValueError:
        return False

def parse_modelfile(multiline_input):
    lines = multiline_input.split('\n')
    filtered_lines = [line for line in lines if not line.startswith('#') and not line.startswith('FROM ') and not line.startswith('failed to get console mode')]
    parsed_output = '\n'.join(filtered_lines)
    return parsed_output

def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value) if not isinstance(value, dict) else 'Invalid value')

def print_status(json_objects):
    lines = json_objects.split('\n')
    for line in lines:
        try:
            data = json.loads(line)
            print(data["status"])
        except json.JSONDecodeError:
            continue

def interrupt_handler(signum, frame):
    print(f"\n\nModel upload aborted, exiting")
    sys.exit(0)

signal.signal(signal.SIGINT, interrupt_handler)

separator = get_platform_separator()

ollama_models = get_env_var("OLLAMA_MODELS", "*")
base_dir = Path(get_platform_path(ollama_models))

if not base_dir.is_dir():
    print(f"Error: ollama models directory ({base_dir}) does not exist.")
    sys.exit(1)

if not validate_url(args.remote_server):
    print(f"Error: remote server URL is not valid: {args.remote_server}")
    sys.exit(1)

blob_dir = Path(f'{base_dir}{separator}blobs')
model_dir = Path(f'{base_dir}{separator}manifests{separator}{args.local_model}')
manifest_file = args.local_model.replace(':', f"{separator}")

if model_base(args.local_model) == "hub":
    model_dir = Path(f'{base_dir}{separator}manifests{separator}{manifest_file}')
elif model_base(args.local_model) == "":
    model_dir = Path(f'{base_dir}{separator}manifests{separator}registry.ollama.ai{separator}library{separator}{manifest_file}')
else:
    model_dir = Path(f'{base_dir}{separator}manifests{separator}registry.ollama.ai{separator}{manifest_file}')

if not model_dir.is_file():
    print(f"Error: model not found in {model_dir}.")
    sys.exit(1)

with open(model_dir, 'r') as mfile:
    data = json.load(mfile)

print(f"Copying model {args.local_model} to {args.remote_server}...")

model_from = ''

for layer in data.get('layers', []):
    if layer.get('mediaType').startswith('application/vnd.ollama.image.model') or layer.get('mediaType').startswith('application/vnd.ollama.image.projector') or layer.get('mediaType').startswith('application/vnd.ollama.image.adapter'):
        digest = layer.get('digest')
        hash = digest[7:]
        try:
            r = requests.head(
                f"{args.remote_server}/api/blobs/sha256:{hash}",
            )
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            sys.exit(1)
        remote_path="@"
        if r.ok:
            print(f"skipping upload for already created layer sha256:{hash}")
        else:
            print(f"uploading layer sha256:{hash}")
            blob_file = f"{blob_dir}{separator}sha256{get_digest_separator()}{hash}"
            with open(blob_file, "r+b") as f:
                total_size = int(os.fstat(f.fileno()).st_size)
                block_size = 1024
                with tqdm(desc="uploading", total=total_size, unit="B", unit_scale=True, unit_divisor=block_size) as progress_bar:
                    wrapped_file = CallbackIOWrapper(progress_bar.update, f, "read")
                    try:
                        r = requests.post(f"{args.remote_server}/api/blobs/sha256:{hash}", data=wrapped_file)
                    except requests.exceptions.RequestException as e:
                        print(f"Error: {e}")
                        sys.exit(1)
                if r.status_code == 201:
                    print("success uploading layer.")
                elif r.status_code == 400:
                    print("Error: invalid digest, check both ollama are running the same version.")
                    sys.exit(1)
                else:
                    print(f"Error: upload failed: {r.reason}")
                    sys.exit(1)
        model_from += f'FROM {remote_path}sha256:{hash}\n'

try:
    result = subprocess.run(["ollama", "show", f"{args.local_model}", "--modelfile"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, encoding='UTF-8', shell=False, check=True)
    if result.stdout.startswith("Error:"):
        print(f"Error: could not get ollama Modelfile")    
    modelfile = parse_modelfile(result.stdout)
    modelfile = model_from + modelfile
except Exception as e:
    print(f"Error: could not run ollama to export Modelfile")
    sys.exit(1)

try:
    headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
    }

    model_create = {
        "name": args.local_model,
        "modelfile": modelfile
    }
    data = json.dumps(model_create)

    try:
        r = requests.post(f"{args.remote_server}/api/create", headers=headers, data=data)
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        sys.exit(1)
    if r.status_code == 200:
        print_status(r.text)
        sys.exit(0)
    else:
        print(f"Error: could not create {args.local_model} on the remote server ({r.status_code}): {r.reason}")
        sys.exit(1)
except Exception as e:
    print(f"Exception: could not create {args.local_model} on the remote server: {e}")
    sys.exit(1)
