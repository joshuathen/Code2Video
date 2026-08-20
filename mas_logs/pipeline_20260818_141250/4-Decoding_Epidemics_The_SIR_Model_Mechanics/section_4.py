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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["R-naught predicts if outbreaks propagate.", "R-naught is the infection-recovery ratio.", "R-naught over one means epidemic growth."]
        self.setup_layout("Critical Thresholds: The R₀ Concept", lecture_lines)
        
        # Setup visual objects
        r_naught_label = MathTex(r"R_0 = \frac{\beta}{\gamma}").set_color(BLUE)
        self.place_at_grid(r_naught_label, "B2")
        
        # Animation stage objects
        infector = Dot(color=RED)
        self.place_at_grid(infector, "D2")
        
        target_group = VGroup(*[Dot(color=GREEN) for _ in range(3)])
        self.place_in_area(target_group, "D4", "E6")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(r_naught_label))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        self.play(FadeIn(infector), FadeIn(target_group))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(RED))
        lines = VGroup(*[Line(infector.get_center(), dot.get_center(), color=RED) for dot in target_group])
        self.play(Create(lines))
        self.wait(2)
