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
        self.setup_layout("Prerequisite Warm-up: Scaling Dimensions", [
            "We study geometry beyond three dimensions.",
            "Distance in n-dimensions follows the Pythagorean theorem.",
            "The sphere equation generalizes as sum of squares."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Create a 2D plane and transform it to 3D
        plane = Square(side_length=2, color="#FFFFFF")
        self.place_at_grid(plane, "C2")
        self.play(Create(plane))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        
        cube = Cube(side_length=1.5, fill_opacity=0.5, color="#FFFF00")
        self.place_at_grid(cube, "C5")
        self.play(ReplacementTransform(plane, cube))
        self.play(self.lecture[0].animate.set_color("#FFFF00"))

        # === Animation for Lecture Line 2 ===
        # Distance formula sqrt(x^2 + y^2) -> sqrt(x^2 + y^2 + z^2)
        dist_2d = MathTex(r"\sqrt{x^2 + y^2}", color="#00FF00")
        dist_3d = MathTex(r"\sqrt{x^2 + y^2 + z^2}", color="#00FF00")
        
        self.place_at_grid(dist_2d, "E2")
        self.play(Write(dist_2d))
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        self.place_at_grid(dist_3d, "E5")
        self.play(ReplacementTransform(dist_2d, dist_3d))

        # === Animation for Lecture Line 3 ===
        # Display sphere equation and generalize with asset
        sphere_eq = MathTex(r"x^2+y^2+z^2=r^2", color="#FF00FF")
        gen_eq = MathTex(r"\sum_{i=1}^{n} x_i^2 = r^2", color="#FF00FF")
        
        self.place_at_grid(sphere_eq, "B3")
        self.play(Write(sphere_eq))
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        
        # Asset integration
        sphere_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        self.place_at_grid(sphere_icon, "B6", scale_factor=0.5)
        self.play(FadeIn(sphere_icon))
        
        self.place_at_grid(gen_eq, "B4")
        self.play(ReplacementTransform(sphere_eq, gen_eq))
        self.wait(2)
