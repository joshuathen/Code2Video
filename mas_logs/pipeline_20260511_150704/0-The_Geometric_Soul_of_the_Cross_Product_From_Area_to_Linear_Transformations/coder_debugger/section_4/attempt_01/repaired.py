from manim import *

class Section4Scene(Scene):
    def construct(self):
        # 1. Title setup
        title = Text("Area and the Determinant", font_size=40)
        title.to_edge(UP)

        # 2. Matrix representation
        # Using standard Matrix class for robustness in CE v0.19.0
        matrix = Matrix([["a", "b"], ["c", "d"]])
        matrix.shift(LEFT * 3)

        # 3. Geometric Formula
        formula = MathTex(r"\text{Area} = |ad - bc|")
        formula.next_to(matrix, RIGHT, buff=1.5)

        # 4. Visual square to represent transformed unit area
        square = Square(side_length=2, color=BLUE, fill_opacity=0.4)
        square.next_to(formula, DOWN, buff=0.5)
        
        # Labels for the matrix
        matrix_label = MathTex("M =").next_to(matrix, LEFT)

        # 5. Animation Sequence
        self.play(Write(title))
        self.wait(0.5)
        
        self.play(
            Create(matrix),
            Write(matrix_label)
        )
        self.wait(0.5)
        
        self.play(Write(formula))
        self.play(DrawBorderThenFill(square))
        
        # Highlight the transformation concept
        self.play(
            square.animate.set_fill(YELLOW, opacity=0.6),
            formula.animate.scale(1.2).set_color(YELLOW)
        )
        
        self.wait(2)