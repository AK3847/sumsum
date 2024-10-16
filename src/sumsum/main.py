import click
import os
import requests
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.console import Console
from rich.live import Live
import ollama

console = Console()

model_dir = os.path.join(os.path.expanduser("~"), ".ollama", "local_summarization")
model_path = os.path.join(model_dir, "Llama_3.2_3B_fine_tune_summarization.gguf")
modelfile_path = os.path.join(model_dir, "ModelFile")
model_name = "sum_model"


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

    click.echo("Searching Model on Ollama server....")
    model_list = ollama.list()
    model_list = model_list["models"]
    for i in range(len(model_list)):
        if f"{model_name}:latest" == model_list[i]["name"]:
            click.echo(
                "Model already exists on Ollama server\nUse 'sumsum run' for summarization "
            )
            return

    click.echo(f"Model {model_name} is not available on Ollama server")
    with open(modelfile_path, "r") as f:
        modelfile = f.read()
    try:
        with Live(
            console.status(
                "[bold green]Integrating Model with Ollama server...[/bold green]"
            )
        ):
            ollama.create(model=model_name, modelfile=modelfile)
        click.echo(
            f"Model {model_name} succesfully integrated with Ollama server!\nUse 'sumsum run' for summarization"
        )
    except Exception as e:
        click.echo(f"Couldn't integrate the model with Ollama Server due to error: {e}")


@cli.command()
@click.argument("text_file", type=click.Path(exists=True))
@click.option(
    "--verbose",
    default=False,
    is_flag=True,
    show_default=True,
    help="To show Additional Information",
)
def run(text_file, verbose):
    """
    Summarize text from a text_file\n
    Sends the text to Ollama sever via api and prints the response
    """

    with open(text_file, "r") as f:
        prompt = f.read()

    with Live(
        console.status("[bold green]Generating response...[/bold green]"),
        refresh_per_second=20,
    ):
        response = ollama.chat(
            model=f"{model_name}", messages=[{"role": "user", "content": f"{prompt}"}]
        )

    click.echo("Response from Model:\n")
    click.echo(response["message"]["content"])
    click.echo("\n")

    if verbose:
        click.echo("Additional Information:\n")
        click.echo(f"Model Load Duration: {int(response['load_duration'])/1e9:.2f}s")
        click.echo(
            f"Total Response Duration: {int(response['total_duration'])/1e9:.2f}s"
        )
        click.echo(f"Tokens Generated: {int(response['eval_count'])}")


def main():
    cli()
