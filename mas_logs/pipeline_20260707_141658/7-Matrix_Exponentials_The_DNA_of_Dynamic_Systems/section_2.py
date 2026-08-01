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

class Section2Scene(TeachingScene):
    def construct(self):
        # Data from shared state
        title_text = "Prerequisite: The Taylor Series Bridge"
        lecture_lines = [
            "- The Taylor series is the blueprint for exponential functions.",
            "- We can plug square matrices into this infinite sum.",
            "- Replace the scalar one with the identity matrix I."
        ]

        # Initialize layout
        self.setup_layout(title_text, lecture_lines)
        
        # Initial state: Dim all lecture lines to start
        for line in self.lecture:
            line.set_opacity(0.3)

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.play(self.lecture[0].animate.set_opacity(1), run_time=1)
        
        # Blueprint: e^x = 1 + x + x^2/2! + ...
        # Formula uses Text for robustness against LaTeX environment issues.
        scalar_formula = Text("e^x = 1 + x + x^2/2! + ...", font_size=36, color=WHITE)
        self.place_in_area(scalar_formula, 'B2', 'B5', scale_factor=0.7) # Issue 25 Fix: Centering formula
        
        self.play(Write(scalar_formula))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Highlight second lecture line and change its color to Yellow
        self.play(
            self.lecture[0].animate.set_opacity(0.3),
            self.lecture[1].animate.set_opacity(1).set_color(YELLOW),
            run_time=1
        )
        
        # Plug in matrix A: e^A = 1 + A + A^2/2! + ...
        # Using t2c (text to color) to highlight 'A' in YELLOW to match lecture line theme
        matrix_formula_A = Text(
            "e^A = 1 + A + A^2/2! + ...", 
            font_size=36, 
            t2c={'A': YELLOW}
        )
        self.place_in_area(matrix_formula_A, 'B2', 'B5', scale_factor=0.7)
        
        self.play(ReplacementTransform(scalar_formula, matrix_formula_A))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Highlight third lecture line and change its color to Emerald Green (#50C878)
        self.play(
            self.lecture[1].animate.set_opacity(0.3),
            self.lecture[2].animate.set_opacity(1).set_color("#50C878"),
            run_time=1
        )
        
        # Replace 1 with I: e^A = I + A + A^2/2! + ...
        # Matrix I is colored Emerald Green as requested by storyboard.
        matrix_formula_I = Text(
            "e^A = I + A + A^2/2! + ...",
            font_size=36,
            t2c={'I': "#50C878", 'A': YELLOW}
        )
        self.place_in_area(matrix_formula_I, 'B2', 'B5', scale_factor=0.7)
        
        self.play(ReplacementTransform(matrix_formula_A, matrix_formula_I))
        self.wait(1.5)

        # === Closing Context (Addressing Issues 26 & 27) ===
        # Adding system eq and solution to bridge to dynamic systems and fulfill critic requests.
        # These objects are placed in the recommended grid positions to balance the layout.
        system_eq = Text("dx/dt = Ax", color=BLUE, font_size=40)
        self.place_at_grid(system_eq, 'D4', scale_factor=0.9) # Issue 26 Fix: Horizontal balancing
        
        solution = Text("x(t) = e^At x(0)", color=GREEN, font_size=40)
        self.place_at_grid(solution, 'E4', scale_factor=0.9) # Issue 27 Fix: Artifact reduction/centering
        
        self.play(FadeIn(system_eq, shift=UP))
        self.wait(0.5)
        self.play(Write(solution))
        self.wait(3)
