import unittest
import os
import tempfile
from src.data_ingestion.latex_parser import parse_latex_file
from src import config # To ensure config paths are initialized if needed by parser indirectly

class TestLatexParser(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test files if it doesn't exist
        self.test_dir = tempfile.mkdtemp()
        # Override RAW_LATEX_DIR for testing if parser uses it directly (it doesn't currently)
        # For now, we pass file paths directly.

    def tearDown(self):
        # Clean up the temporary directory and its contents
        for root, dirs, files in os.walk(self.test_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.test_dir)

    def create_temp_tex_file(self, content, filename="test.tex"):
        file_path = os.path.join(self.test_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return file_path

    def test_parse_simple_latex(self):
        content = """
\\documentclass{article}
% This is a comment.
\\title{Test Title}
\\begin{document}
Hello World.
This is a test sentence.
An inline formula $E=mc^2$.
\\section{First Section}
Some text in section 1.
Another formula: \\[ \\alpha + \\beta = \\gamma \\]
\\end{document}
        """
        file_path = self.create_temp_tex_file(content)
        parsed_text = parse_latex_file(file_path)

        self.assertIn("Hello World.", parsed_text)
        self.assertIn("This is a test sentence.", parsed_text)
        self.assertIn("$E=mc^2$", parsed_text, "Inline math should be preserved")
        self.assertNotIn("% This is a comment.", parsed_text, "Comments should be removed")
        self.assertIn("\\section{First Section}", parsed_text, "Section command should be preserved")
        self.assertIn("\\[ \\alpha + \\beta = \\gamma \\]", parsed_text, "Display math should be preserved")
        self.assertNotIn("\\title{Test Title}", parsed_text, "Title macro (and others not explicitly handled) should be removed by current custom_latex_to_text")
        self.assertNotIn("\\documentclass{article}", parsed_text) # Should be removed
        self.assertNotIn("\\begin{document}", parsed_text) # Should be removed
        self.assertNotIn("\\end{document}", parsed_text) # Should be removed

    def test_empty_file(self):
        file_path = self.create_temp_tex_file("")
        parsed_text = parse_latex_file(file_path)
        self.assertEqual(parsed_text, "")

    def test_file_not_found(self):
        parsed_text = parse_latex_file("non_existent_file.tex")
        self.assertEqual(parsed_text, "") # Expect empty string on error

    def test_math_environments(self):
        content = """
\\begin{document}
Inline: $a^2+b^2=c^2$.
Display: \\[ x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a} \\]
Equation:
\\begin{equation}
\\nabla \\cdot \\mathbf{E} = \\frac{\\rho}{\\epsilon_0}
\\end{equation}
Align:
\\begin{align}
f(x) &= (x+a)(x+b) \\\\
     &= x^2 + (a+b)x + ab
\\end{align}
\\end{document}
        """
        file_path = self.create_temp_tex_file(content)
        parsed_text = parse_latex_file(file_path)
        self.assertIn("$a^2+b^2=c^2$", parsed_text)
        self.assertIn("\\[ x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a} \\]", parsed_text)
        # pylatexenc's LatexMathNode might not preserve \begin{equation} verbatim by default with latex_verbatim()
        # It depends on how it's tokenized. The current custom_latex_to_text aims to keep the verbatim math.
        # Let's check for the core math content instead of exact environment tags if issues arise.
        self.assertIn("\\nabla \\cdot \\mathbf{E} = \\frac{\\rho}{\\epsilon_0}", parsed_text) # Check core content
        self.assertIn("f(x) &= (x+a)(x+b)", parsed_text) # Check core content

    def test_structural_commands(self):
        content = """
\\documentclass{article}
\\begin{document}
\\section{Main Section}
Text here.
\\subsection{Subsection A}
More text.
\\subsubsection{Subsubsection B}
Details.
\\paragraph{A Paragraph Title}
Paragraph content.
\\end{document}
        """
        file_path = self.create_temp_tex_file(content)
        parsed_text = parse_latex_file(file_path)
        self.assertIn("\\section{Main Section}", parsed_text)
        self.assertIn("\\subsection{Subsection A}", parsed_text)
        self.assertIn("\\subsubsection{Subsubsection B}", parsed_text)
        self.assertIn("\\paragraph{A Paragraph Title}", parsed_text)

if __name__ == '__main__':
    unittest.main()
