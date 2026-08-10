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
        lecture_lines = [
            "Mass ratio impacts grow with magnitude.",
            "Number of collisions converges to pi.",
            "Dynamics bridge physics and mathematics.",
            "Precision increases with heavier blocks.",
            "Limits reveal the circle constant."
        ]
        self.setup_layout("The Limit of Infinity (100^n)", lecture_lines)
        
        # Load Assets
        blocks_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg")
        
        # === Animation for Lecture Line 1 ===
        n_expr = MathTex("100^n", color=WHITE)
        self.place_at_grid(n_expr, "B2", scale_factor=1.5)
        self.place_at_grid(blocks_asset, "B4", scale_factor=0.5)
        self.play(self.lecture[0].animate.set_color("#FF0000"), n_expr.animate.set_color("#FF0000"), Indicate(blocks_asset))

        # === Animation for Lecture Line 2 ===
        collision_count = MathTex(r"\text{Collisions} \to \pi", color=WHITE)
        self.place_at_grid(collision_count, "C2", scale_factor=1.2)
        self.play(self.lecture[1].animate.set_color("#FFFF00"), Write(collision_count))

        # === Animation for Lecture Line 3 ===
        arc = Arc(radius=0.5, start_angle=0, angle=PI, color=WHITE)
        self.place_at_grid(arc, "D2", scale_factor=1.0)
        self.play(self.lecture[2].animate.set_color("#00FFFF"), Create(arc))

        # === Animation for Lecture Line 4 ===
        digits = MathTex("3.14159...", color=WHITE)
        self.place_at_grid(digits, "E2", scale_factor=1.2)
        self.play(self.lecture[3].animate.set_color("#00FF00"), Write(digits))

        # === Animation for Lecture Line 5 ===
        pi_val = MathTex(r"\pi \approx 3.14159265", color=WHITE)
        self.place_at_grid(pi_val, "F2", scale_factor=1.2)
        final_blocks = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg")
        self.place_at_grid(final_blocks, "F5", scale_factor=0.5)
        self.play(self.lecture[4].animate.set_color("#FFFFFF"), Write(pi_val), FadeIn(final_blocks))
        self.wait(2)
