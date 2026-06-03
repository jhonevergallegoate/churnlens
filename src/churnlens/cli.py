"""CLI principal del proyecto ChurnLens.

Construida con `typer`, expone las operaciones del paquete como
sub-comandos legibles y bien documentados. Útil tanto para uso humano
como para automatización en CI/CD.

Comandos disponibles:

```
churnlens info
churnlens data download         [--force]
churnlens data validate
churnlens data summary
churnlens data materialize
churnlens data hash
churnlens preprocess run        [--no-engineered]
churnlens eda report
churnlens features select       [--k 20]
churnlens model train           [--model NAME] [--cv 5] [--use-consensus]
churnlens model evaluate        [--model NAME] [--split val|test] [--threshold 0.5]
churnlens model list
churnlens serve                 [--host 127.0.0.1] [--port 8000] [--workers 1] [--reload]
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
preprocess_app = typer.Typer(
    help="Preprocesamiento + feature engineering + split (Fase 2).",
    no_args_is_help=True,
)
eda_app = typer.Typer(help="Análisis exploratorio reproducible (Fase 2).", no_args_is_help=True)
features_app = typer.Typer(
    help="Selección y extracción de características (Fase 3).",
    no_args_is_help=True,
)
model_app = typer.Typer(
    help="Entrenamiento, evaluación y registro de modelos (Fase 3).",
    no_args_is_help=True,
)
app.add_typer(data_app, name="data")
app.add_typer(preprocess_app, name="preprocess")
app.add_typer(eda_app, name="eda")
app.add_typer(features_app, name="features")
app.add_typer(model_app, name="model")

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
# Sub-comandos `preprocess`
# ---------------------------------------------------------------------------
@preprocess_app.command("run")
def cmd_preprocess_run(
    no_engineered: Annotated[
        bool,
        typer.Option(
            "--no-engineered",
            help="Omite la generación de features derivadas.",
        ),
    ] = False,
) -> None:
    """Ejecuta el pipeline completo de preprocesamiento de Fase 2."""
    from churnlens.features.pipeline import run_preprocessing

    try:
        artifacts = run_preprocessing(include_engineered=not no_engineered)
    except Exception as exc:  # pragma: no cover - defensa CLI
        log.error("preprocess_failed", error=str(exc))
        console.print(f"[red]✗ Preprocesamiento fallido:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="ChurnLens · Artefactos de preprocesamiento", show_header=False)
    table.add_column("Archivo", style="bold cyan")
    table.add_column("Ruta")
    for name, path in artifacts.to_dict().items():
        table.add_row(name, path)
    console.print(table)


# ---------------------------------------------------------------------------
# Sub-comandos `eda`
# ---------------------------------------------------------------------------
@eda_app.command("report")
def cmd_eda_report() -> None:
    """Genera figuras y tablas del análisis exploratorio."""
    from churnlens.eda.report import generate_eda_report

    try:
        report = generate_eda_report()
    except Exception as exc:  # pragma: no cover - defensa CLI
        log.error("eda_failed", error=str(exc))
        console.print(f"[red]✗ EDA fallido:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="ChurnLens · Artefactos EDA", show_header=True)
    table.add_column("Tipo", style="bold cyan")
    table.add_column("Nombre", style="bold")
    table.add_column("Ruta")
    for name, path in report.figures.items():
        table.add_row("figura", name, str(path))
    for name, path in report.tables.items():
        table.add_row("tabla", name, str(path))
    console.print(table)


# ---------------------------------------------------------------------------
# Sub-comandos `features`
# ---------------------------------------------------------------------------
@features_app.command("select")
def cmd_features_select(
    k: Annotated[int, typer.Option("--k", help="Tamaño del consenso top-k.")] = 20,
    rf_estimators: Annotated[
        int, typer.Option("--rf-estimators", help="Árboles del RF en permutation importance.")
    ] = 200,
    permutation_repeats: Annotated[
        int, typer.Option("--permutation-repeats", help="Repeticiones del shuffle.")
    ] = 10,
) -> None:
    """Ejecuta las cuatro técnicas de selección y persiste el consenso."""
    from churnlens.features.selection import persist_feature_selection, run_feature_selection

    try:
        result = run_feature_selection(
            k=k,
            rf_estimators=rf_estimators,
            permutation_repeats=permutation_repeats,
        )
        paths = persist_feature_selection(result)
    except Exception as exc:  # pragma: no cover - defensa CLI
        log.error("feature_selection_failed", error=str(exc))
        console.print(f"[red]✗ Selección fallida:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title=f"ChurnLens · Top-{k} features por consenso", show_header=True)
    table.add_column("#", justify="right", style="bold cyan")
    table.add_column("Feature", style="bold")
    table.add_column("Votos", justify="right")
    table.add_column("Score medio (norm.)", justify="right")
    for i, row in result.consensus.head(k).iterrows():
        table.add_row(
            str(int(i) + 1),
            str(row["feature"]),
            str(int(row["votes"])),
            f"{row['mean_norm_score']:.4f}",
        )
    console.print(table)

    paths_table = Table(title="Artefactos", show_header=False)
    paths_table.add_column("Archivo", style="bold cyan")
    paths_table.add_column("Ruta")
    for name, p in paths.items():
        paths_table.add_row(name, str(p))
    console.print(paths_table)


# ---------------------------------------------------------------------------
# Sub-comandos `model`
# ---------------------------------------------------------------------------
@model_app.command("train")
def cmd_model_train(
    model_names: Annotated[
        list[str] | None,
        typer.Option(
            "--model",
            "-m",
            help="Modelo a entrenar (puede repetirse). Default = todos.",
        ),
    ] = None,
    cv: Annotated[int, typer.Option("--cv", help="Folds de validación cruzada.")] = 5,
    use_consensus: Annotated[
        bool,
        typer.Option(
            "--use-consensus",
            help="Restringe X al top-k del consenso de selección.",
        ),
    ] = False,
    baselines_only: Annotated[
        bool,
        typer.Option(
            "--baselines-only",
            help="Atajo: entrena solo dummies + logreg balanced.",
        ),
    ] = False,
) -> None:
    """Entrena los modelos solicitados, registra artefactos y reporte."""
    from churnlens.models.baseline import BASELINE_MODEL_NAMES
    from churnlens.models.train import train_models

    feature_subset: list[str] | None = None
    if use_consensus:
        manifest_path = settings.processed_dir / "feature_consensus.json"
        if not manifest_path.exists():
            console.print(
                f"[red]✗ No existe {manifest_path}. Ejecuta primero "
                "`churnlens features select`.[/red]"
            )
            raise typer.Exit(code=1)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        feature_subset = list(manifest.get("top_k_features") or [])

    if baselines_only:
        model_names = list(BASELINE_MODEL_NAMES)

    try:
        artifacts = train_models(
            models=model_names,
            feature_subset=feature_subset,
            cv=cv,
        )
    except Exception as exc:  # pragma: no cover - defensa CLI
        log.error("model_train_failed", error=str(exc))
        console.print(f"[red]✗ Entrenamiento fallido:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="ChurnLens · Comparativa de modelos (val)", show_header=True)
    cols = (
        "model",
        "val_pr_auc",
        "cv_pr_auc_mean",
        "val_roc_auc",
        "val_f1_tuned",
        "val_threshold_tuned",
    )
    for c in cols:
        table.add_column(
            c,
            justify="right" if c != "model" else "left",
            style="bold cyan" if c == "model" else "",
        )
    for _, row in artifacts.summary_table.iterrows():
        table.add_row(
            *[
                str(row["model"]),
                f"{row['val_pr_auc']:.4f}",
                f"{row['cv_pr_auc_mean']:.4f}",
                f"{row['val_roc_auc']:.4f}",
                f"{row['val_f1_tuned']:.4f}",
                f"{row['val_threshold_tuned']:.3f}",
            ]
        )
    console.print(table)
    console.print(f"[green]✓ Ganador por PR-AUC: [bold]{artifacts.best_model_name}[/bold][/green]")
    for name, p in artifacts.paths.items():
        console.print(f"  {name}: {p}")


@model_app.command("evaluate")
def cmd_model_evaluate(
    model_name: Annotated[
        str,
        typer.Option("--model", "-m", help="Nombre del modelo registrado."),
    ],
    split: Annotated[
        str,
        typer.Option("--split", help="'val' o 'test'."),
    ] = "val",
    threshold: Annotated[
        float | None,
        typer.Option("--threshold", help="Threshold; default = el sintonizado del manifest."),
    ] = None,
) -> None:
    """Evalúa un modelo registrado en el split indicado."""
    from churnlens.models.evaluation import binary_metrics
    from churnlens.models.registry import load_model

    try:
        model, metadata = load_model(model_name)
    except FileNotFoundError as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if split not in ("val", "test"):
        console.print("[red]--split debe ser 'val' o 'test'.[/red]")
        raise typer.Exit(code=1)

    parquet_path = settings.processed_dir / f"{split}.parquet"
    if not parquet_path.exists():
        console.print(f"[red]No existe {parquet_path}. Ejecuta el preprocesamiento.[/red]")
        raise typer.Exit(code=1)

    import pandas as pd

    from churnlens.features.preprocessing import TARGET_COL
    from churnlens.models.train import _predict_proba

    df = pd.read_parquet(parquet_path)
    y = df[TARGET_COL].astype("int8").to_numpy()
    feature_set = metadata.get("feature_set") or [c for c in df.columns if c != TARGET_COL]
    x = df[feature_set].astype("float32").to_numpy()

    proba = _predict_proba(model, x)
    if threshold is None:
        threshold = float(
            ((metadata.get("metrics") or {}).get("val_tuned") or {}).get("threshold", 0.5)
        )
    metrics = binary_metrics(y, proba, threshold=threshold)

    table = Table(title=f"ChurnLens · {model_name} sobre {split}", show_header=False)
    table.add_column("Métrica", style="bold cyan")
    table.add_column("Valor", justify="right")
    for k, v in metrics.items():
        table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    console.print(table)


@model_app.command("list")
def cmd_model_list() -> None:
    """Lista los modelos registrados en `models/`."""
    from churnlens.models.registry import list_models

    entries = list_models()
    if not entries:
        console.print("[yellow]No hay modelos registrados todavía.[/yellow]")
        return

    table = Table(title="ChurnLens · Modelos registrados", show_header=True)
    for col in ("name", "algorithm", "val_pr_auc", "val_f1_tuned", "created_at"):
        table.add_column(col, style="bold cyan" if col == "name" else "")
    for entry in entries:
        metrics_val = (entry.metrics.get("val") or {}) if isinstance(entry.metrics, dict) else {}
        metrics_tuned = (
            (entry.metrics.get("val_tuned") or {}) if isinstance(entry.metrics, dict) else {}
        )
        table.add_row(
            entry.name,
            entry.algorithm,
            f"{metrics_val.get('pr_auc', float('nan')):.4f}",
            f"{metrics_tuned.get('f1', float('nan')):.4f}",
            entry.created_at,
        )
    console.print(table)


# ---------------------------------------------------------------------------
# Comando `serve` (Fase 4)
# ---------------------------------------------------------------------------
@app.command()
def serve(
    host: Annotated[
        str,
        typer.Option("--host", help="Interfaz de red donde escucha la API."),
    ] = settings.api_host,
    port: Annotated[
        int,
        typer.Option("--port", help="Puerto HTTP de la API."),
    ] = settings.api_port,
    workers: Annotated[
        int,
        typer.Option("--workers", help="Número de procesos uvicorn (producción)."),
    ] = 1,
    reload: Annotated[
        bool,
        typer.Option("--reload", help="Auto-reload al cambiar código (solo desarrollo)."),
    ] = False,
) -> None:
    """Levanta la API de inferencia FastAPI (Fase 4)."""
    import uvicorn

    console.print(
        f"[green]✓[/green] ChurnLens API en [bold]http://{host}:{port}[/bold] "
        f"(docs en [bold]http://{host}:{port}/docs[/bold])"
    )
    uvicorn.run(
        "churnlens.serving.api:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point usado por el script `churnlens` instalado por setuptools."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
