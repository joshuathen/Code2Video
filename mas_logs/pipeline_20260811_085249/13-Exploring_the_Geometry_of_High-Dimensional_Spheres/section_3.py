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
        self.setup_layout("The Paradox: Where is the Volume?", [
            "Volume concentrates near the surface.",
            "Inner core becomes empty.",
            "Paradox: volume approaches zero."
        ])
        
        # Elements using assets
        # Load assets once
        outer_sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        inner_core = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/core.svg")
        surface_shell = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/surface.svg")
        
        # Position using area to satisfy constraints
        self.place_in_area(outer_sphere, "B4", "E6", scale_factor=1.0)
        outer_sphere.set_color("#8A2BE2")

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(outer_sphere))
        self.lecture[0].set_color("#8A2BE2")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Visualize density shifting
        self.place_in_area(inner_core, "B4", "E6", scale_factor=0.6)
        inner_core.set_color("#FF4500")
        
        self.place_in_area(surface_shell, "B4", "E6", scale_factor=1.0)
        surface_shell.set_color("#FF4500")

        self.play(FadeIn(inner_core), FadeIn(surface_shell))
        self.lecture[1].set_color("#FF4500")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Shrink the inner core volume to zero graphically
        self.play(
            inner_core.animate.scale(0.01), 
            FadeOut(surface_shell), 
            FadeOut(outer_sphere)
        )
        self.lecture[2].set_color(WHITE)
        self.wait(2)
