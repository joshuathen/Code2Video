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
        self.setup_layout("Defining Eigenvectors and Eigenvalues", [
            "Eigenvectors are vectors that keep their original span.",
            "They only scale by an eigenvalue 'lambda'.",
            "This is defined by the equation Av = lambda v."
        ])
        
        # Define objects
        formula = MathTex("A", "v", "=", "\\lambda", "v")
        vec_v = Vector(RIGHT * 1.5, color=GREEN)
        label_v = Text("v", color=GREEN).next_to(vec_v, UP)
        label_lambda = Text("\\lambda", color=YELLOW)
        
        # Position initial objects based on feedback
        self.place_in_area(formula, 'B3', 'B5', scale_factor=1.2)
        self.place_at_grid(vec_v, 'C3', scale_factor=0.9)
        self.place_at_grid(label_v, 'C2', scale_factor=0.9) # Adjusting label_v next to vec_v

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(GREEN))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.place_at_grid(label_lambda, 'C4', scale_factor=0.9)
        self.play(Write(label_lambda))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.play(FadeIn(formula))
        self.wait(2)
