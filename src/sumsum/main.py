import click
import os
import requests
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn


model_dir = os.path.join(os.path.expanduser("~"), ".ollama", "local_summarization")
model_path = os.path.join(model_dir, "Llama_3.2_3B_fine_tune_summarization.gguf")
modelfile_path = os.path.join(model_dir, "ModelFile")


def download_model():
    url = "https://huggingface.co/AKT47/Llama_3.2_3B_fine_tune_summarization/resolve/main/unsloth.Q8_0.gguf"

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
                TimeRemainingColumn(),
            )
            task = progress.add_task("Downloading", total=total_size)

            with progress:
                with open(model_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=2048):
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))
        click.echo(f"Model downlaoded succesfully and saved as {model_path}")
    except Exception as e:
        click.echo(f"An error occured while downloading: {e}")


def generate_model_file():
    url = "https://huggingface.co/AKT47/Llama_3.2_3B_fine_tune_summarization/resolve/main/Modelfile"
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(modelfile_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=2048):
                    f.write(chunk)

        with open(modelfile_path, "r") as f:
            lines = f.readlines()

        lines[0] = ""
        lines[1] = f"FROM {model_path}\n"

        with open(modelfile_path, "w") as f:
            f.writelines(lines)

        click.echo("ModelFile Generated!")
    except Exception as e:
        click.echo(f"An erro occurred while creating the ModelFile: {e}")


@click.group()
def cli():
    """Sum-Sum : Local summarization"""


@cli.command()
def help():
    """Show Options and functions"""


@cli.command()
def init():
    """
    Initialize environment
    - Check for Ollama
    - Download the gguf file
    -
    """

    click.echo("Checking for Ollama......")
    if os.system("ollama --version") == 0:
        click.echo("Ollama is installed!")
    else:
        click.echo(
            """Ollama is not installed! \nPlease install the latest version from: https://ollama.com/download"""
        )
        return

    click.echo("Checking if model is already downloaded or not...")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if os.path.exists(model_path):
        click.echo(f"Model already downloaded at {model_path}")
    else:
        click.echo("Downloading Model......")
        download_model()

    click.echo("Checking for Modelfile....")
    if os.path.exists(modelfile_path):
        click.echo(f"ModelFile already exists at {modelfile_path}.")
        if click.confirm(
            "Do you want to overwrite the existing ModelFile?", default=False
        ):
            click.echo("Overwriting ModelFile...")
            generate_model_file()
        else:
            click.echo("Using the existing ModelFile.")
    else:
        click.echo(f"ModelFile not found!\n Generating Modelfile at {modelfile_path}.")
        generate_model_file()


def main():
    cli()
