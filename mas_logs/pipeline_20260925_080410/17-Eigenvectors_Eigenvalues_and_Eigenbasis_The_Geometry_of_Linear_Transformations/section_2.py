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
        lecture_lines = [
            "An eigenvector stays in its direction.",
            "The matrix maps it to a multiple.",
            "That multiplier is the eigenvalue.",
            "It represents the scaling factor.",
            "Equation: A times v equals lambda v."
        ]
        self.setup_layout("Defining Eigenvectors and Eigenvalues", lecture_lines)
        
        # Elements
        vector_v = Vector([1.5, 1, 0], color=BLUE).shift(self.grid['C3'])
        vector_av = Vector([3, 2, 0], color=BLUE).shift(self.grid['C3'])
        
        # Prepare labels
        label_v = MathTex("v", color=BLUE)
        label_av = MathTex("Av", color=BLUE)
        label_lambda = MathTex("\\lambda", color=YELLOW)
        equation = MathTex("A", "v", "=", "\\lambda", "v")
        equation.set_color_by_tex("v", BLUE)
        equation.set_color_by_tex("\\lambda", YELLOW)
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(vector_v), Write(self.place_at_grid(label_v, 'B2')))
        self.lecture[0].set_color(BLUE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(Transform(vector_v.copy(), vector_av), Write(self.place_at_grid(label_av, 'B4')))
        self.lecture[1].set_color(BLUE)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(Write(self.place_at_grid(label_lambda, 'D4')))
        self.lecture[2].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(vector_v.animate.scale(2), run_time=1)
        self.lecture[3].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(FadeOut(vector_v, vector_av, label_v, label_av, label_lambda),
                  Write(self.place_in_area(equation, 'B2', 'E5')))
        self.lecture[4].set_color(WHITE)
        self.wait(2)
