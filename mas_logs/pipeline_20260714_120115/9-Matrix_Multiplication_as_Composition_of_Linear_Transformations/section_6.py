from manim import *

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

class Section6Scene(TeachingScene):
    def construct(self):
        # Initialize layout with title and lecture lines
        self.setup_layout("Summary: Matrix Multiplication is Combining Actions", [
            "Matrix multiplication is the \"glue\" for linear actions.",
            "It combines multiple transformations into a single operation.",
            "Complex motions are built from simple matrix products."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        
        # Show a chain of matrix multiplications: M3 * M2 * M1 * v.
        # Use Text instead of MathTex for stability.
        formula = Text("M3 · M2 · M1 · v", color="#FFFF00", font_size=32)
        # Wide formula uses rectangular area (L015)
        self.place_in_area(formula, "B2", "B6", scale_factor=0.9)
        
        self.play(Write(formula))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Reset previous line color and highlight current
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FF8C00")
        )
        
        # Briefly flash three distinct transformations on a single grid in rapid succession.
        grid_lines = VGroup(
            Line(LEFT*2.5, RIGHT*2.5, color=GREY_E),
            Line(UP*1.5, DOWN*1.5, color=GREY_E)
        )
        # Expanded area to utilize more space (Fix 42)
        self.place_in_area(grid_lines, "C2", "E6")
        
        square = Square(side_length=1.4, color="#FF8C00", stroke_width=3)
        # Expanded area to avoid cramped appearance (Fix 43)
        self.place_in_area(square, "C2", "E6")
        
        self.play(Create(grid_lines), FadeIn(square))
        
        # Transformation 1: Rotation
        self.play(square.animate.rotate(PI/3), run_time=0.6)
        # Transformation 2: Shear (Linear Action)
        self.play(square.animate.apply_matrix([[1, 0.5], [0, 1]]), run_time=0.6)
        # Transformation 3: Scaling
        self.play(square.animate.scale(0.5), run_time=0.6)
        
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Reset previous line color and highlight current
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#00FFFF")
        )
        
        # End with the text "Matrices = Composition" in a bright color (#00FFFF).
        summary_text = Text("Matrices = Composition", color="#00FFFF", font_size=32)
        # Bottom area utilized (L017)
        self.place_in_area(summary_text, "F2", "F6", scale_factor=0.9)
        
        self.play(Write(summary_text))
        self.wait(3)
