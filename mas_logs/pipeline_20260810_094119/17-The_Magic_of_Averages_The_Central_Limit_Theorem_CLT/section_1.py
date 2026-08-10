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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Individual events often seem like random chaos.",
            "But averages reveal a hidden, perfect order.",
            "Watch 1,000 dice rolls aggregate together.",
            "They form a beautiful, stable bell curve.",
            "Nature loves these consistent patterns."
        ]
        self.setup_layout("The Hook: Chaos to Order", lecture_lines)
        
        # Load asset
        dice_svg = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/dice.svg"
        
        # === Animation for Lecture Line 1 ===
        # Create chaotic cloud
        dots = VGroup(*[SVGMobject(dice_svg, color="#FF5733").scale(0.05) for _ in range(50)])
        self.place_in_area(dots, 'B4', 'F6', scale_factor=0.5)
        self.play(FadeIn(dots))
        self.lecture[0].set_color("#FF5733")

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#3498DB")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(dots.animate.arrange_in_grid(rows=5, cols=10, buff=0.1).scale(0.4).move_to(self.grid["D4"]))
        self.lecture[2].set_color("#3498DB")

        # === Animation for Lecture Line 4 ===
        # Bell Curve
        curve = FunctionGraph(lambda x: 1.5 * np.exp(-x**2), x_range=[-3, 3], color="#3498DB")
        self.place_in_area(curve, 'D2', 'F5', scale_factor=0.6)
        self.play(ReplacementTransform(dots, curve))
        
        # Highlight center
        peak = Dot(color=WHITE).move_to(curve.point_from_proportion(0.5))
        circle = Circle(radius=0.3, color=WHITE).move_to(peak)
        self.play(Create(circle))
        self.lecture[3].set_color("#3498DB")

        # === Animation for Lecture Line 5 ===
        order_text = Text("Order from Chaos", font_size=30, color=WHITE)
        self.place_at_grid(order_text, 'D3', scale_factor=0.9)
        self.play(Write(order_text))
        self.lecture[4].set_color("#FFFFFF")
        self.wait(2)
