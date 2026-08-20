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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Different light wavelengths rotate at different rates.",
            "White light fans out creating color gradients.",
            "Deeper travel maps a candy-cane helical twist.",
            "Increasing path length makes the twist distinct.",
            "The effect resembles a rotating barber pole."
        ]
        self.setup_layout("Mechanism: The Barber Pole Effect", lecture_lines)

        # Animation Elements
        # 1. Plane-polarized light vector & Asset
        vector = Arrow(ORIGIN, UP * 1.5, color=WHITE)
        barberpole_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/barberpole.svg")
        self.place_at_grid(vector, "E2", scale_factor=0.6)
        self.place_at_grid(barberpole_asset, "E2", scale_factor=0.4)

        # 2. Sugar solution container & Asset
        container = Rectangle(width=2, height=4, color=BLUE).set_fill(BLUE, opacity=0.3)
        sugar_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sugar.svg")
        self.place_at_grid(container, "C4", scale_factor=1)
        self.place_at_grid(sugar_asset, "C4", scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(Create(vector), FadeIn(barberpole_asset))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        # Indicate color fanning out
        dot_group = VGroup(*[Dot(color=color) for color in [RED, GREEN, BLUE]])
        dot_group.arrange(RIGHT)
        self.place_at_grid(dot_group, "B6", scale_factor=0.7)
        self.play(FadeIn(dot_group))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFCC00"))
        # Helical path representation
        helix = ParametricFunction(
            lambda t: np.array([0, t/2, np.sin(t*2)*0.5]),
            t_range=np.array([0, 4]),
            color=YELLOW
        )
        self.place_at_grid(helix, "D4", scale_factor=0.6)
        self.play(Create(helix), FadeIn(sugar_asset))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF6666"))
        self.play(helix.animate.stretch(1.5, dim=1))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF00FF"))
        # Rotate vector to represent barber pole effect
        self.play(Rotate(vector, angle=PI/2, about_point=vector.get_start()))
