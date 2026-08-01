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
        # Initialize Scene
        title = "Prerequisite: The Key Factory (Hash Chains)"
        lines = [
            "Privacy begins with a secure key hierarchy.",
            "A secret seed generates a unique daily key.",
            "This daily key produces many rotating identifiers."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_SK = "#FFD700"
        COLOR_DK = "#00FF00"
        COLOR_RPI = "#00BFFF"

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color(COLOR_SK))

        # Secret Key (SK) Asset and Label
        sk_asset = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/key.svg", color=COLOR_SK)
        sk_box = Rectangle(width=2.5, height=1.2, color=COLOR_SK)
        sk_label = Text("Secret Key (SK)", font_size=18, color=COLOR_SK).next_to(sk_box, UP, buff=0.1)
        sk_group = VGroup(sk_box, sk_asset, sk_label)
        
        self.place_in_area(sk_group, "A2", "B5", scale_factor=0.8)
        self.play(FadeIn(sk_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second line
        self.play(self.lecture[1].animate.set_color(COLOR_DK))

        # Daily Key (DK)
        dk_box = Rectangle(width=2.5, height=1.0, color=COLOR_DK)
        dk_label = Text("Daily Key (DK)", font_size=18, color=COLOR_DK).next_to(dk_box, UP, buff=0.1)
        dk_group = VGroup(dk_box, dk_label)
        self.place_in_area(dk_group, "C2", "D5", scale_factor=0.8)

        # Hash Process
        hash_text = Text("H()", font_size=24, color=WHITE)
        self.place_at_grid(hash_text, "C3", scale_factor=0.8)
        # Adjust hash position to be between boxes
        hash_text.move_to((self.grid["B3"] + self.grid["C3"]) / 2)

        arrow_sk_to_hash = Arrow(sk_box.get_bottom(), hash_text.get_top(), color=WHITE, buff=0.1)
        arrow_hash_to_dk = Arrow(hash_text.get_bottom(), dk_box.get_top(), color=WHITE, buff=0.1)

        self.play(Create(arrow_sk_to_hash), Write(hash_text))
        self.play(Create(arrow_hash_to_dk), FadeIn(dk_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third line
        self.play(self.lecture[2].animate.set_color(COLOR_RPI))

        # RPI Circles
        rpi_positions = ["E1", "E2", "E3", "E4", "E5", "E6"]
        rpi_circles = VGroup()
        rpi_arrows = VGroup()

        for pos in rpi_positions:
            circle = Circle(radius=0.3, color=COLOR_RPI, fill_opacity=0.3)
            label = Text("RPI", font_size=12, color=COLOR_RPI).move_to(circle.get_center())
            rpi_unit = VGroup(circle, label)
            self.place_at_grid(rpi_unit, pos, scale_factor=0.7)
            rpi_circles.add(rpi_unit)
            
            arrow = Arrow(dk_box.get_bottom(), rpi_unit.get_top(), color=COLOR_RPI, stroke_width=2, buff=0.05)
            rpi_arrows.add(arrow)

        self.play(
            LaggedStart(*[Create(arr) for arr in rpi_arrows], lag_ratio=0.1),
            LaggedStart(*[FadeIn(circ) for circ in rpi_circles], lag_ratio=0.1),
            run_time=2
        )
        self.wait(2)
