import unittest
import os
import tempfile
import re # Added import for regular expressions
from src.data_ingestion.latex_parser import parse_latex_file
from src import config # To ensure config paths are initialized if needed by parser indirectly

class TestLatexParser(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()

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
\\author{An Author}
\\date{\\today}
\\begin{document}
\\maketitle
Hello World.
This is a test sentence.
An inline formula $E=mc^2$.
\\section{First Section}
Some text in section 1.
Another formula: \\[ \\alpha + \\beta = \\gamma \\]
\\subsection{A Subsection}
More text.
\\end{document}
        """
        file_path = self.create_temp_tex_file(content)
        parsed_text = parse_latex_file(file_path)

        # Print parsed text for debugging if tests fail
        # print("\n--- Parsed text for test_parse_simple_latex ---")
        # print(parsed_text)
        # print("--- End of parsed text ---\n")

        self.assertIn("Hello World.", parsed_text)
        self.assertIn("This is a test sentence.", parsed_text)
        self.assertIn("$E=mc^2$", parsed_text, "Inline math should be preserved")
        self.assertNotIn("% This is a comment.", parsed_text, "Comments should be removed")
        
        # Updated assertions for default section handling
        self.assertIn("FIRST SECTION", parsed_text, "Section title text should be present")
        self.assertNotIn("\\section{First Section}", parsed_text, "Raw section command should be converted")
        self.assertIn("§", parsed_text, "Default section marker '§' should be present if that's the default output")


        self.assertIn("\\[ \\alpha + \\beta = \\gamma \\]", parsed_text, "Display math should be preserved")
        
        # Macros like \title, \author, \maketitle are in the `macros_to_remove` list in latex_parser.py
        # AND will now have explicit MacroTextSpec to output ""
        self.assertNotIn("Test Title", parsed_text, "Title content should be removed")
        self.assertNotIn("An Author", parsed_text, "Author content should be removed")
        # The date might still appear if \today is expanded by a default handler before \date is removed.
        # For now, let's focus on title and author.
        
        self.assertNotIn("\\documentclass{article}", parsed_text)

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
        self.assertIn("\\begin{equation}\n\\nabla \\cdot \\mathbf{E} = \\frac{\\rho}{\\epsilon_0}\n\\end{equation}", parsed_text)
        self.assertIn("\\begin{align}\nf(x) &= (x+a)(x+b) \\\\\n     &= x^2 + (a+b)x + ab\n\\end{align}", parsed_text)


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

        self.assertIn("MAIN SECTION", parsed_text)
        self.assertNotIn("\\section{Main Section}", parsed_text)
        self.assertIn("§ MAIN SECTION", parsed_text) 

        self.assertIn("Subsection A", parsed_text)
        self.assertNotIn("\\subsection{Subsection A}", parsed_text)
        self.assertIn("§.§ Subsection A", parsed_text) 

        self.assertIn("Subsubsection B", parsed_text)
        self.assertNotIn("\\subsubsection{Subsubsection B}", parsed_text)
        self.assertIn("§.§.§ Subsubsection B", parsed_text)

        self.assertIn("A Paragraph Title", parsed_text)
        self.assertIn("Paragraph content.", parsed_text)
        self.assertNotIn("\\paragraph{A Paragraph Title}", parsed_text)
        self.assertTrue(re.search(r"A Paragraph Title\s*\n\s*Paragraph content.", parsed_text, re.IGNORECASE),
                        "Paragraph title and content format not as expected.")


if __name__ == '__main__':
    # import re # No longer needed here as it's at the top level
    unittest.main()