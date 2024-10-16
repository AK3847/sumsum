import click
import os
import requests
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.console import Console
from rich.live import Live
import ollama
from rich.prompt import Confirm

console = Console()

model_dir = os.path.join(os.path.expanduser("~"), ".ollama", "local_summarization")
model_path = os.path.join(model_dir, "Llama_3.2_3B_fine_tune_summarization.gguf")
modelfile_path = os.path.join(model_dir, "ModelFile")
model_name = "sum_model"


rich_color_teal = "555555"
rich_color_green = "00f200"
rich_color_red = "a7232e"
rich_color_white = "bfbfbf"
rich_color_turqoise = "11a8cd"
rich_color_yellow = "f2f200"


def download_model():
    url = "https://huggingface.co/AKT47/Llama_3.2_3B_fine_tune_summarization/resolve/main/unsloth.Q8_0.gguf"

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(
                    bar_width=None,
                    complete_style=f"#{rich_color_red}",
                    finished_style=f"#{rich_color_green}",
                ),
                TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
                TimeRemainingColumn(),
            )
            task = progress.add_task(
                f"[#{rich_color_turqoise}]Downloading", total=total_size
            )

            with progress:
                with open(model_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=2048):
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))
        console.print(
            f"[#{rich_color_teal}]Model downlaoded succesfully and saved as [italic #{rich_color_white}]{model_path}"
        )
    except Exception as e:
        console.print(f"[#{rich_color_red}]An error occured while downloading: {e}")


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

    console.print(f"[#{rich_color_teal}]Checking for Ollama.....")
    if os.system("ollama --version >nul 2>&1") == 0:
        console.print(f"[#{rich_color_green}]Ollama is installed!")
    else:
        console.print(
            f"[#{rich_color_red}]Ollama is not installed!\nPlease install the latest version for your OS from: [#{rich_color_white}]Please install the latest version from: https://ollama.com/download"
        )
        return

    console.print(
        f"[#{rich_color_teal}]Checking if model is already downloaded or not....."
    )
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if os.path.exists(model_path):
        console.print(
            f"[#{rich_color_green}]Model already downloaded at [italic #{rich_color_white}]{model_path}"
        )
    else:
        console.print(f"[#{rich_color_teal}Downloading Model.....]")
        download_model()

    console.print(f"[#{rich_color_teal}]Checking for Modelfile.....")
    if os.path.exists(modelfile_path):
        console.print(
            f"[#{rich_color_green}]ModelFile already exists at [italic #{rich_color_white}]{modelfile_path}."
        )
        if Confirm.ask(
            f"[#{rich_color_teal}]Do you want to overwrite the existing ModelFile?",
            show_choices=True,
            default=False,
            show_default=False,
        ):
            console.print(f"[#{rich_color_teal}]Overwriting ModelFile.....")
            generate_model_file()
        else:
            console.print(f"[#{rich_color_teal}]Using the existing ModelFile.")
    else:
        console.print(
            f"[#{rich_color_red}]ModelFile not found!\n Generating Modelfile at {modelfile_path}."
        )
        generate_model_file()

    console.print(f"[#{rich_color_teal}]Searching Model on Ollama server.....")
    model_list = ollama.list()
    model_list = model_list["models"]
    for i in range(len(model_list)):
        if f"{model_name}:latest" == model_list[i]["name"]:
            console.print(
                f"[#{rich_color_green}]Model [italic bold #{rich_color_white}]{model_name}[/italic bold #{rich_color_white}] already exists on Ollama server!"
            )
            console.print(
                f"[#{rich_color_green}]Use [italic bold #{rich_color_white}]'sumsum run'[/italic bold #{rich_color_white}] for summarization"
            )
            return

    console.print(
        f"[#{rich_color_red}]Model [italic bold #{rich_color_white}]{model_name}[/italic bold #{rich_color_white}] is not available on Ollama server"
    )

    with open(modelfile_path, "r") as f:
        modelfile = f.read()
    try:
        with Live(
            console.status(
                f"[#{rich_color_turqoise}]Integrating Model with Ollama server....."
            ),
            transient=True,
        ):
            ollama.create(model=model_name, modelfile=modelfile)
        console.print(
            f"[#{rich_color_green}]Model [italic bold #{rich_color_white}]{model_name}[/italic bold #{rich_color_white}] succesfully integrated with Ollama server!"
        )
        console.print(
            f"[#{rich_color_green}]Use [italic bold #{rich_color_white}]'sumsum run'[/italic bold #{rich_color_white}] for summarization "
        )
    except Exception as e:
        console.print(
            f"[#{rich_color_red}]Couldn't integrate the model with Ollama Server due to error: {e}"
        )


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
        console.status(
            f"[#{rich_color_turqoise}]Generating response...[/#{rich_color_turqoise}]"
        ),
        refresh_per_second=20,
        transient=True,
    ):
        response = ollama.chat(
            model=f"{model_name}",
            messages=[
                {
                    "role": "system",
                    "content": "Summarize the given user content as precise as possible",
                },
                {"role": "user", "content": f"{prompt}"},
            ],
        )

    console.print(f"[#{rich_color_yellow}]Response from model:\n")
    console.print(f"[#{rich_color_white}]{response['message']['content']}\n")

    if verbose:
        console.print(f"[#{rich_color_yellow}]Additional Information:\n")
        console.print(
            f"[#{rich_color_white}]Model load duration: {int(response['load_duration'])/1e9:.2f}s"
        )
        console.print(
            f"[#{rich_color_white}]Total response duration: {int(response['load_duration'])/1e9:.2f}s"
        )
        console.print(
            f"[#{rich_color_white}]Tokens generated: {int(response['eval_count'])}"
        )
        console.print(
            f"[#{rich_color_white}]Speed of generation (Tokens/second): {int(response['eval_count'])/(int(response['eval_duration'])/1e9):.2f}"
        )


def main():
    cli()
