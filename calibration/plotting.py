import logging
import pandas as pd
from pathlib import Path
from meval.diags import rel_diag


def render_reliability_plots(
    test_df: pd.DataFrame, output_dir: Path, prefix: str
) -> None:
    """
    Generates and saves reliability diagrams for original and recalibrated predictions.
    """
    plot_df = test_df[test_df["drain"].isin([0.0, 1.0])].copy()
    if plot_df.empty:
        logging.warning("Skipping reliability plot for %s: no drain in {0,1}", prefix)
        return
    plot_df["drain"] = plot_df["drain"].astype(int)
    plot_df["label"] = plot_df["label"].astype(bool)

    for variant, score_col, title in (
        ("original", "y_prob", "Original reliability by drain group"),
        (
            "recalibrated",
            "y_prob_calibrated",
            "Recalibrated reliability by drain group",
        ),
    ):
        variant_df = plot_df[["label", "drain", score_col]].rename(
            columns={score_col: "y_prob"}
        )
        fig, _, _ = rel_diag(
            variant_df,
            plot_groups=["drain=0", "drain=1"],
            fig_title=title,
            legend=True,
            add_risk_density=True,
            threshold=float(variant_df["label"].mean()),
        )
        html_path = output_dir / f"{prefix}_{variant}_reliability.html"
        png_path = output_dir / f"{prefix}_{variant}_reliability.png"
        fig.write_html(str(html_path))
        try:
            fig.write_image(str(png_path))
        except Exception as error:
            logging.warning("PNG export skipped for %s (%s)", png_path, error)
