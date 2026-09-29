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
        self.setup_layout("Synthesis and Summary", [
            "Invertibility means space is fully preserved.",
            "Singular matrices collapse dimensions, losing information.",
            "Reflecting on mirror analogy for synthesis."
        ])

        # === Animation for Lecture Line 1 ===
        axes = ThreeDAxes(x_range=[-2, 2], y_range=[-2, 2], z_range=[-2, 2], axis_config={"include_tip": True})
        self.place_in_area(axes, "A1", "C3", scale_factor=0.6)
        label_full = Text("Full Space", font_size=20, color=BLUE)
        self.place_at_grid(label_full, "A3")
        self.add(axes, label_full)
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        flattened_grid = VGroup(*[Line(start=[-1, -1, 0], end=[1, 1, 0], color=RED) for _ in range(5)])
        self.place_in_area(flattened_grid, "D1", "F3", scale_factor=0.5)
        label_lost = Text("Information Lost", font_size=20, color=RED)
        self.place_at_grid(label_lost, "F3")
        self.add(flattened_grid, label_lost)
        self.lecture[1].set_color(RED)

        # === Animation for Lecture Line 3 ===
        mirror_icon = Square(color=YELLOW).scale(0.5)
        self.place_at_grid(mirror_icon, "B5", scale_factor=0.8)
        label_mirror = Text("Mirror", font_size=20, color=YELLOW)
        label_distortion = Text("Distortion", font_size=20, color=YELLOW)
        self.place_at_grid(label_mirror, "A5")
        self.place_at_grid(label_distortion, "C5")
        self.add(mirror_icon, label_mirror, label_distortion)
        self.lecture[2].set_color(YELLOW)
        self.wait()
