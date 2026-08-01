from manim import *

class Section4Scene(Scene):
    def construct(self):
        # 1. Title setup
        title = Text("Area and the Determinant", font_size=40)
        title.to_edge(UP)

        # 2. Matrix representation
        # Using Text for elements to avoid LaTeX dependency (FileNotFoundError: 'latex')
        # Note: Matrix brackets in Manim CE v0.19.0 also rely on LaTeX; 
        # for environments without LaTeX, using VGroup with Text is a robust fallback.
        matrix = VGroup(
            Text("[ a  b ]"),
            Text("[ c  d ]")
        ).arrange(DOWN, buff=0.2)
        matrix.shift(LEFT * 3)

        # 3. Geometric Formula
        formula = Text("Area = |ad - bc|")
        formula.next_to(matrix, RIGHT, buff=1.5)

        # 4. Visual square to represent transformed unit area
        square = Square(side_length=2, color=BLUE, fill_opacity=0.4)
        square.next_to(formula, DOWN, buff=0.5)
        
        # Labels for the matrix
        matrix_label = Text("M =").next_to(matrix, LEFT)

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
        self