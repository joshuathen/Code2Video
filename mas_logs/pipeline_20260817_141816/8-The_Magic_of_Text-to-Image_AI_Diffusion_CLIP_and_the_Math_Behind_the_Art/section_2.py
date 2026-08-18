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
        lecture_lines = [
            "Computers see images as high-dimensional vectors.",
            "Latent space is a map of concepts.",
            "Similar concepts cluster together in space."
        ]
        self.setup_layout("The Concept of Latent Space", lecture_lines)
        
        # Define objects
        latent_plane = NumberPlane(
            x_range=(-3, 3, 1),
            y_range=(-3, 3, 1),
            background_line_style={"stroke_color": GREY, "stroke_width": 1}
        )
        # Apply fix for issues 25/26 (overlap of area)
        self.place_in_area(latent_plane, 'C2', 'F5', scale_factor=0.6)
        
        # Define clusters (dots)
        dots = VGroup(
            Dot(color="#00FFFF").move_to(latent_plane.c2p(1, 1)),
            Dot(color="#00FFFF").move_to(latent_plane.c2p(1.2, 0.8)),
            Dot(color="#00FFFF").move_to(latent_plane.c2p(0.8, 1.2)),
            Dot(color="#FF00FF").move_to(latent_plane.c2p(-1, -1)),
            Dot(color="#FF00FF").move_to(latent_plane.c2p(-1.2, -0.8))
        )
        
        # Integrate Asset
        computer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        self.place_at_grid(computer_icon, 'B2', scale_factor=0.5)
        
        # Apply fix for issue 24 (label position)
        label = Text("Latent Space", color=WHITE, font_size=24)
        self.place_at_grid(label, 'A4', scale_factor=0.8)
        
        # Moving point
        moving_dot = Dot(color=YELLOW).move_to(latent_plane.c2p(-2, 2))

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        self.play(FadeIn(computer_icon), Create(latent_plane), FadeIn(dots))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFFFF")
        self.play(Write(label))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF00FF")
        self.add(moving_dot)
        self.play(moving_dot.animate.move_to(latent_plane.c2p(1.1, 1.1)), run_time=2)
        self.wait(1)
