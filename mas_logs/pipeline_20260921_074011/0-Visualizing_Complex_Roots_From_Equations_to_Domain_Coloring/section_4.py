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
        self.setup_layout("Domain Coloring: Painting the Complex Plane", [
            "Domain coloring reveals hidden structures.",
            "Map phase to hue, magnitude to light.",
            "Colors expose roots as bright centers."
        ])
        
        # Color definitions
        color1 = "#FFFF00"
        color2 = "#FF6666"
        color3 = "#77FF77"
        color4 = "#E0E0E0"

        # Assets
        palette_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/palette.svg")
        brush_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/brush.svg")
        lantern_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lantern.svg")

        # Objects
        # Color Wheel representative
        circle = Circle(radius=1.0, color=WHITE)
        wedge_group = VGroup()
        for i in range(8):
            wedge = Sector(angle=PI/4, start_angle=i*PI/4, color=interpolate_color(BLUE, RED, i/8))
            wedge_group.add(wedge)
        color_wheel = VGroup(wedge_group, circle, palette_icon)

        # Root visualization dummy
        root_center = VGroup(Dot(color=WHITE, radius=0.1), lantern_icon)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(color1)
        self.place_at_grid(color_wheel, "B2", scale_factor=0.6)
        self.play(FadeIn(color_wheel))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(color2)
        # Using brush asset as indicator
        self.place_at_grid(brush_icon, "C4", scale_factor=0.5)
        self.play(FadeIn(brush_icon))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(color3)
        self.place_at_grid(root_center, "E3", scale_factor=1.0)
        self.play(FadeIn(root_center))
        self.play(Indicate(root_center, color=color4))
        self.wait(2)
