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
        title = "Application: The Central Limit Theorem Preview"
        lines = [
            "Adding more variables continuously smooths the resulting shape.",
            "Two uniform distributions convolve into a triangle shape.",
            "Adding a third variable creates a bell-like curve.",
            "Repeated convolution leads toward the Normal distribution.",
            "This is the essence of the Central Limit Theorem."
        ]
        self.setup_layout(title, lines)

        # Colors
        GREEN = "#00FF00"
        BLUE = "#00BFFF"
        YELLOW = "#FFFF00"
        RED = "#FF0000"
        GOLD = "#FFD700"
        WHITE = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(GREEN)
        
        sq1 = Polygon(
            [-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0],
            color=GREEN
        ).set_fill(color=GREEN, opacity=0.5)
        
        sq2 = Polygon(
            [-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0],
            color=BLUE
        ).set_fill(color=BLUE, opacity=0.5)
        
        conv_sym = MathTex("*", color=GOLD, font_size=48)
        
        self.place_at_grid(sq1, "B2", scale_factor=0.8)
        self.place_at_grid(conv_sym, "B3", scale_factor=1.0)
        self.place_at_grid(sq2, "B4", scale_factor=0.8)

        self.play(Create(sq1), Create(sq2), Write(conv_sym))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        
        triangle = Polygon(
            [-1, -0.5, 0], [0, 0.5, 0], [1, -0.5, 0],
            color=YELLOW
        ).set_fill(color=YELLOW, opacity=0.5)
        
        # Issue 36 Fix: Move triangle to area C2-D3 to avoid overlap with conv_sym2
        self.place_in_area(triangle, "C2", "D3", scale_factor=0.8)
        
        self.play(
            ReplacementTransform(VGroup(sq1, sq2).copy(), triangle)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(RED)
        
        sq3 = Polygon(
            [-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0],
            color=BLUE
        ).set_fill(color=BLUE, opacity=0.5)
        
        # Issue 37 Fix: Vertical alignment for sq3 and conv_sym2
        self.place_in_area(sq3, "C5", "D5", scale_factor=0.8)
        
        conv_sym2 = MathTex("*", color=GOLD, font_size=48)
        self.place_in_area(conv_sym2, "C4", "D4", scale_factor=1.0)
        
        # Irwin-Hall n=3 curve (bell-like)
        # Using a simple Gaussian-like curve for visualization
        bell_curve = FunctionGraph(
            lambda x: 1.2 * np.exp(-(x**2)),
            x_range=[-1.5, 1.5],
            color=RED
        ).set_stroke(width=4)
        self.place_in_area(bell_curve, "E2", "F4", scale_factor=0.8)
        
        self.play(FadeIn(sq3), FadeIn(conv_sym2))
        self.wait(0.5)
        
        self.play(ReplacementTransform(VGroup(triangle, sq3).copy(), bell_curve))
        self.wait(1.5)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(WHITE)
        
        normal_label = Text("Normal Distribution", color=WHITE, font_size=20)
        # Issue 38 Fix: Reduce scale to 0.5 to avoid screen clipping
        self.place_at_grid(normal_label, "F5", scale_factor=0.5)

        self.play(Write(normal_label))
        self.wait(1.5)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(GOLD)
        
        # L004 Fix: Use Indicate instead of Indication (though code was already using Indicate)
        self.play(
            Indicate(conv_sym, color=GOLD, scale_factor=1.5),
            Indicate(conv_sym2, color=GOLD, scale_factor=1.5)
        )
        self.wait(2.0)
