"""CLI principal del proyecto ChurnLens.

Construida con `typer`, expone las operaciones del paquete como
sub-comandos legibles y bien documentados. Útil tanto para uso humano
como para automatización en CI/CD.

Comandos disponibles:

```
churnlens info
churnlens data download   [--force]
churnlens data validate
churnlens data summary
churnlens data materialize
churnlens data hash
```
"""

from __future__ import annotations

import json
import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from churnlens import __author__, __email__, __version__
from churnlens.config import settings
from churnlens.data.loader import TelcoChurnLoader
from churnlens.logger import get_logger

app = typer.Typer(
    name="churnlens",
    help="ChurnLens — Predicción temprana de churn (Diplomado MLDS · UNAL).",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
data_app = typer.Typer(help="Operaciones sobre el dataset.", no_args_is_help=True)
app.add_typer(data_app, name="data")

console = Console()
log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Comandos raíz
# ---------------------------------------------------------------------------
@app.command()
def info() -> None:
    """Imprime metadatos del paquete y de la configuración activa."""
    table = Table(title="ChurnLens · Información del entorno", show_header=False)
    table.add_column("Campo", style="bold cyan")
    table.add_column("Valor")
    table.add_row("Versión", __version__)
    table.add_row("Autor", f"{__author__} <{__email__}>")
    table.add_row("Proyecto raíz", str(settings.project_root))
    table.add_row("Data URL", settings.data_url)
    table.add_row("Raw CSV", str(settings.raw_csv_path))
    table.add_row("Interim Parquet", str(settings.interim_parquet_path))
    table.add_row("Log level", settings.log_level)
    table.add_row("Random seed", str(settings.random_seed))
    console.print(table)


# ---------------------------------------------------------------------------
# Sub-comandos `data`
# ---------------------------------------------------------------------------
@data_app.command("download")
def cmd_download(
    force: Annotated[
        bool, typer.Option("--force", help="Re-descarga aunque el archivo ya exista.")
    ] = False,
) -> None:
    """Descarga el dataset crudo a `data/raw/`."""
    loader = TelcoChurnLoader()
    try:
        path = loader.download(force=force)
    except Exception as exc:  # pragma: no cover - defensa CLI
        log.error("download_failed", error=str(exc))
        raise typer.Exit(code=1) from exc
    console.print(f"[green]✓[/green] Dataset descargado en [bold]{path}[/bold]")


@data_app.command("validate")
def cmd_validate() -> None:
    """Valida el dataset contra el esquema Pandera."""
    loader = TelcoChurnLoader()
    try:
        df = loader.load_validated()
    except Exception as exc:
        log.error("validation_failed", error=str(exc))
        console.print(f"[red]✗ Validación fallida:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[green]✓[/green] Esquema válido — "
        f"[bold]{len(df):,}[/bold] filas × [bold]{df.shape[1]}[/bold] columnas."
    )


@data_app.command("materialize")
def cmd_materialize() -> None:
    """Materializa el dataset validado a parquet en `data/interim/`."""
    loader = TelcoChurnLoader()
    path = loader.materialize_interim()
    console.print(f"[green]✓[/green] Parquet materializado en [bold]{path}[/bold]")


@data_app.command("summary")
def cmd_summary(
    as_json: Annotated[bool, typer.Option("--json", help="Emite la salida como JSON.")] = False,
) -> None:
    """Imprime un resumen ejecutivo del dataset."""
    loader = TelcoChurnLoader()
    summary = loader.summary()

    if as_json:
        json.dump(summary.to_dict(), sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return

    table = Table(title="ChurnLens · Resumen del dataset", show_header=False)
    table.add_column("Métrica", style="bold cyan")
    table.add_column("Valor", justify="right")
    table.add_row("Filas", f"{summary.n_rows:,}")
    table.add_row("Columnas", f"{summary.n_cols}")
    table.add_row("Tasa de churn (positivos)", f"{summary.target_pos_rate:.4f}")
    table.add_row("NaN en TotalCharges", f"{summary.n_missing_total_charges}")
    table.add_row("Tamaño (bytes)", f"{summary.bytes:,}")
    if summary.md5:
        table.add_row("MD5", summary.md5)
    if summary.sha256:
        table.add_row("SHA-256", summary.sha256)
    console.print(table)


@data_app.command("hash")
def cmd_hash() -> None:
    """Imprime los hashes MD5/SHA-256 registrados del archivo crudo."""
    from churnlens.utils.hashing import load_checksums  # local import

    records = load_checksums(settings.checksums_path)
    if not records:
        console.print(
            "[yellow]No hay checksums registrados. "
            "Ejecuta primero `churnlens data download`.[/yellow]"
        )
        raise typer.Exit(code=1)
    json.dump(records, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point usado por el script `churnlens` instalado por setuptools."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
