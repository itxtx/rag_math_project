import pytest
import os
from unittest.mock import patch, mock_open, MagicMock
from src.data_ingestion import latex_processor
import subprocess

def test_latex_with_uncommon_packages(tmp_path):
    uncommon_latex = r"""
    \documentclass{article}
    \usepackage{tikz, xcolor, amssymb, minted}
    \begin{document}
    Hello with uncommon packages.
    \end{document}
    """
    # Patch open to simulate preamble file
    with patch("builtins.open", mock_open(read_data="\\documentclass{article}\n\\usepackage{tikz, xcolor, amssymb, minted}\n\\begin{document}")):
        with patch("os.path.exists", return_value=True):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="<document><body>Uncommon packages parsed</body></document>", stderr="", returncode=0)
                xml = latex_processor.run_latexml_on_content(uncommon_latex)
                assert "Uncommon packages parsed" in xml

def test_latex_with_undefined_command(tmp_path):
    latex_with_error = r"""
    \documentclass{article}
    \begin{document}
    This will fail: \undefinedcommand
    \end{document}
    """
    with patch("builtins.open", mock_open(read_data="\\documentclass{article}\n\\begin{document}")):
        with patch("os.path.exists", return_value=True):
            with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "latexml", output="", stderr="Undefined control sequence")) as mock_run:
                xml = latex_processor.run_latexml_on_content(latex_with_error)
                assert xml == ""

def test_latexml_complex_xml_output():
    complex_xml = """
    <document>
      <body>
        <section>
          <theorem xml:id='thm1'><title>Theorem 1</title><para>Let $x$ be a set.</para></theorem>
          <proof><para>This is a proof.</para></proof>
        </section>
      </body>
    </document>
    """
    # Patch subprocess.run to return this XML
    with patch("builtins.open", mock_open(read_data="\\documentclass{article}\n\\begin{document}")):
        with patch("os.path.exists", return_value=True):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout=complex_xml, stderr="", returncode=0)
                xml = latex_processor.run_latexml_on_content("dummy")
                assert "<theorem" in xml and "<proof>" in xml 