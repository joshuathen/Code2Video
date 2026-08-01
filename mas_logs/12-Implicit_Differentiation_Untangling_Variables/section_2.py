from manim import *

# Set the compiler globally to 'pdflatex', which is standard for Manim environments.
# The error occurred because 'latex' was not found in the system PATH.
config.tex_compiler = "pdflatex"

class Section2Scene(Scene):
    def construct(self):
        # Changed tex_compiler to 'pdflatex' as 'lualatex' was missing in this environment.
        pdflatex_template = TexTemplate(
            tex_compiler="pdflatex",
            output_format=".pdf"
        )

        # Define consistent colors for clarity
        func_color = "#90EE90"  # Light Green for the function/variable
        deriv_color = PINK       # Pink for the derivative component

        # Step 1: Display the General Power Rule using f(x)
        # We split the LaTeX into parts to facilitate TransformMatchingTex and coloring.
        # We pass the custom template to each MathTex instance to ensure the correct compiler is used.
        formula_fx = MathTex(
            r"\frac{d}{dx}[", r"f(x)", r"]^n = n", r"f(x)", r"^{n-1} \cdot", r"f'(x)",
            tex_template=pdflatex_template
        )
        formula_fx.set_color_by_tex(r"f(x)", func_color)
        formula_fx.set_color_by_tex(r"f'(x)", deriv_color)

        self.play(Write(formula_fx))
        self.wait(1)