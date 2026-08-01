from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section5Scene(TeachingScene):
    def construct(self):
        # Data from storyboard
        title = "The Math of Connectivity: Matrix Multiplication"
        lecture_lines = [
            "Layers communicate through massive grids of connections called matrices.",
            "We use matrix multiplication to calculate many neurons at once.",
            "The dot product sums up all weighted inputs simultaneously.",
            "Brightly glowing connections represent high-strength weights in the grid.",
            "This parallel processing makes modern AI fast and efficient."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors
        COLOR_WHITE = "#FFFFFF"
        COLOR_GOLD = "#FFD700"
        COLOR_DIM = GRAY

        # Initial dimming of lecture lines to show progress via highlighting
        self.lecture.set_color(COLOR_DIM)

        # === Animation for Lecture Line 1 ===
        # Display Vector X (Input) and Matrix W (Weights) in white (#FFFFFF).
        self.play(self.lecture[0].animate.set_color(COLOR_WHITE))
        
        # Matrix and Vector values
        w_vals = [["w_{11}", "w_{12}", "w_{13}"], 
                  ["w_{21}", "w_{22}", "w_{23}"], 
                  ["w_{31}", "w_{32}", "w_{33}"]]
        x_vals = [["x_1"], ["x_2"], ["x_3"]]
        
        m_w = Matrix(w_vals).set_color(COLOR_WHITE)
        m_x = Matrix(x_vals).set_color(COLOR_WHITE)
        
        w_label = Text("Matrix W", font_size=20, color=COLOR_WHITE)
        x_label = Text("Vector X", font_size=20, color=COLOR_WHITE)
        
        # Issue 33 Fix: Adjusting positions to avoid overlap with lecture text
        self.place_in_area(m_w, "C3", "E4", scale_factor=0.6)
        self.place_in_area(m_x, "C5", "E6", scale_factor=0.6)
        self.place_at_grid(w_label, "B3", scale_factor=0.8)
        self.place_at_grid(x_label, "B5", scale_factor=0.8)
        
        self.play(FadeIn(m_w), FadeIn(m_x), FadeIn(w_label), FadeIn(x_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # We use matrix multiplication to calculate many neurons at once.
        # Animate dot product: row of W meeting column of X with flash.
        self.play(self.lecture[0].animate.set_color(COLOR_DIM), self.lecture[1].animate.set_color(COLOR_WHITE))
        
        # Highlight first row and column
        row1 = m_w.get_rows()[0]
        col1 = m_x.get_columns()[0]
        
        self.play(Indicate(row1), Indicate(col1))
        
        # Result dot to represent the result of the dot product
        res_dot = Dot(color=COLOR_WHITE)
        self.place_at_grid(res_dot, "D4", scale_factor=1.0)
        
        # Flash animation (L008: using Flash directly)
        self.play(FadeIn(res_dot), Flash(res_dot.get_center(), color=COLOR_WHITE, line_length=0.3, num_lines=8))
        self.wait(1)
        self.play(FadeOut(res_dot))

        # === Animation for Lecture Line 3 ===
        # The dot product sums up all weighted inputs simultaneously.
        # Show a grid of connections between layers glowing by weight strength.
        self.play(self.lecture[1].animate.set_color(COLOR_DIM), self.lecture[2].animate.set_color(COLOR_WHITE))
        
        # Transition to neural network view
        self.play(FadeOut(m_w), FadeOut(m_x), FadeOut(w_label), FadeOut(x_label))
        
        layer1_dots = VGroup(*[Dot(radius=0.1, color=WHITE) for _ in range(4)])
        layer1_dots.arrange(DOWN, buff=0.4)
        layer2_dots = VGroup(*[Dot(radius=0.1, color=WHITE) for _ in range(4)])
        layer2_dots.arrange(DOWN, buff=0.4)
        
        # Issue 34 Fix: Moving dots right to avoid lecture text overlap
        self.place_in_area(layer1_dots, "B3", "E3", scale_factor=0.8)
        self.place_in_area(layer2_dots, "B5", "E5", scale_factor=0.8)
        
        connections = VGroup()
        for d1 in layer1_dots:
            for d2 in layer2_dots:
                # Use set_stroke for opacity as per L008 hints
                line = Line(d1.get_center(), d2.get_center(), stroke_width=1).set_stroke(opacity=0.3)
                connections.add(line)
        
        self.play(Create(layer1_dots), Create(layer2_dots), Create(connections))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Brightly glowing connections represent high-strength weights in the grid.
        # Display equation Y = WX + B in gold (#FFD700).
        self.play(self.lecture[2].animate.set_color(COLOR_DIM), self.lecture[3].animate.set_color(COLOR_GOLD))
        
        # Select some connections to "glow"
        indices = [0, 5, 10, 15]
        glowing_lines = VGroup(*[connections[i].copy().set_stroke(color=COLOR_GOLD, opacity=1.0, width=3) for i in indices])
        
        self.play(Indicate(glowing_lines))
        
        # Equation (L022 fallback to Text if MathTex fails, but usually MathTex is fine in MAS)
        equation = MathTex("Y = WX + B", color=COLOR_GOLD)
        # Issue 35 Fix: Place equation in Row A to avoid vertical overlap
        self.place_in_area(equation, "A3", "A5", scale_factor=1.0)
        
        self.play(Write(equation))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This parallel processing makes modern AI fast and efficient.
        # Animate all output neurons lighting up simultaneously for parallel calc.
        self.play(self.lecture[3].animate.set_color(COLOR_DIM), self.lecture[4].animate.set_color(COLOR_GOLD))
        
        # All output neurons "compute" simultaneously (L004: use Indicate)
        self.play(
            *[Indicate(d, color=COLOR_GOLD) for d in layer2_dots],
            # Small pulse for the equation
            equation.animate.scale(1.1)
        )
        self.play(equation.animate.scale(1/1.1))
        
        self.wait(2)
