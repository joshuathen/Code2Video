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
            "Can we reach the bottom faster?",
            "Not by taking a straight path.",
            "Compare straight line and curve.",
            "A secret path hides beneath.",
            "This is the Brachistochrone problem."
        ]
        self.setup_layout("Introduction: The Challenge of Descent", lecture_lines)
        
        # Assets
        bead_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/bead.svg"
        bead_s = SVGMobject(bead_asset, color=BLUE)
        bead_c = SVGMobject(bead_asset, color=BLUE)
        label_a = Text("A", font_size=24)
        label_b = Text("B", font_size=24)
        
        # Paths
        straight_path = Line(start=self.grid["C3"], end=self.grid["D4"], color=GRAY)
        curve_path = ParametricFunction(
            lambda t: np.array([
                self.grid["C3"][0] + t * (self.grid["D4"][0] - self.grid["C3"][0]),
                self.grid["C3"][1] + (t - t**2) * 1.5,
                0
            ]), 
            t_range=[0, 1], 
            color=GRAY
        )
        
        # Positioning requested by VideoCritic
        self.place_at_grid(bead_s, "C3", scale_factor=0.3)
        self.place_at_grid(bead_c, "D4", scale_factor=0.3)
        self.place_at_grid(label_a, "C2", scale_factor=0.5)
        self.place_at_grid(label_b, "D3", scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        self.play(FadeIn(bead_s), FadeIn(bead_c), Write(label_a), Write(label_b))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FF0000")
        self.play(Create(straight_path))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FF00")
        self.play(Create(curve_path))
        self.play(
            MoveAlongPath(bead_s, straight_path),
            MoveAlongPath(bead_c, curve_path),
            run_time=2
        )

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FF00FF")
        self.play(curve_path.animate.set_stroke(color="#FF00FF", width=6))

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FFFF00")
        self.wait(1)
