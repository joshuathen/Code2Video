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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Mathematical Definition: The Eigen-Equation", [
            "Eigen-equation is Av equals lambda times v.",
            "Lambda represents the scaling factor of elongation.",
            "Det A minus lambda I equals zero."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display the matrix equation A * v = λ * v in white
        equation = MathTex("A", "v", "=", "\\lambda", "v")
        equation[0].set_color("#FFFF00") # A yellow
        equation[1].set_color("#00FF00") # v green
        equation[4].set_color("#00FF00") # v green
        self.place_at_grid(equation, 'B3', scale_factor=1.0)
        self.play(Write(equation))
        self.lecture[0].set_color("#FFFF00")

        # === Animation for Lecture Line 2 ===
        # Lambda represents the scaling factor
        self.play(Indicate(equation[3]))
        self.lecture[1].set_color("#00FF00")

        # === Animation for Lecture Line 3 ===
        # Det A minus lambda I equals zero
        det_eq = MathTex("\\text{det}(A - \\lambda I) = 0")
        self.place_at_grid(det_eq, 'E3', scale_factor=1.0)
        self.play(Write(det_eq))
        self.lecture[2].set_color("#ADD8E6")
        
        self.wait(2)
