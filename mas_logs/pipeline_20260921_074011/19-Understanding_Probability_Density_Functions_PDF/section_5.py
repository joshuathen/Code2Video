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
        self.setup_layout("Summary & Key Takeaways", [
            "Area under the curve equals probability.",
            "Single point probability is always zero.",
            "Normalization ensures total area is one."
        ])
        
        # Define graphics
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 1.5, 0.5], x_length=4, y_length=3, axis_config={"include_tip": False})
        curve = axes.plot(lambda x: np.exp(-(x-2)**2 / 0.5), x_range=[0, 4])
        area = axes.get_area(curve, x_range=[1, 3], color=BLUE, opacity=0.3)
        formula = MathTex(r"P(a < X < b) = \int_{a}^{b} f(x)dx", font_size=28)
        norm_text = MathTex(r"\int_{-\infty}^{\infty} f(x)dx = 1", font_size=32, color=YELLOW)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        plot_group = VGroup(axes, curve, area)
        self.place_at_grid(plot_group, 'C3', scale_factor=0.8)
        self.place_in_area(formula, 'B4', 'C6', scale_factor=0.9)
        self.play(Create(axes), Create(curve), FadeIn(area), Write(formula))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(GREEN)
        point = Dot(axes.c2p(2, 0.8), color=RED)
        point_label = Text("P(X=x) = 0", font_size=20, color=RED)
        self.place_at_grid(point_label, 'B3', scale_factor=0.8)
        self.play(FadeIn(point), Write(point_label))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.play(FadeOut(area), FadeOut(point), FadeOut(point_label), FadeOut(formula))
        self.place_in_area(norm_text, 'D4', 'E6', scale_factor=0.75)
        self.play(Write(norm_text))
        
        self.wait(2)
