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
        self.setup_layout("Introduction to Dandelin Spheres", [
            "Dandelin spheres hide inside the cone.",
            "They touch the cone and the plane.",
            "Tangency points reveal geometric secrets."
        ])

        # Assets
        cone_2d = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg").set_color("#808080").set_opacity(0.8)
        circ1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg").set_color("#ADD8E6")
        circ2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg").set_color("#ADD8E6")
        plane_line = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/plane.svg").set_color("#FF0000")
        foci = VGroup(Dot(color="#FFFF00"), Dot(color="#FFFF00"))

        # Setup positions based on VideoCritic recommendations
        self.place_in_area(cone_2d, 'A4', 'F6', scale_factor=1.2)
        self.place_at_grid(circ1, 'B4', scale_factor=0.7)
        self.place_at_grid(circ2, 'D4', scale_factor=0.7)
        self.place_at_grid(plane_line, 'C4', scale_factor=1.0)
        
        # Animations
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(cone_2d))
        self.play(self.lecture[0].animate.set_color("#ADD8E6"))
        self.play(FadeIn(circ1), FadeIn(circ2))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF0000"))
        self.play(FadeIn(plane_line))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        self.place_at_grid(foci[0], 'B4')
        self.place_at_grid(foci[1], 'D4')
        self.play(FadeIn(foci))
