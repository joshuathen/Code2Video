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
        self.setup_layout("Introducing Dandelin Spheres", [
            "Meet the Dandelin Spheres, a geometric bridge.",
            "They fit perfectly inside the cone's interior.",
            "They are tangent to both cone and plane."
        ])
        
        # Using SVG assets as requested by orchestrator
        cone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg")
        sphere1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        sphere2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        
        # Initial positioning of cone
        self.place_in_area(cone, 'B4', 'E6', scale_factor=1.5)
        
        tangency1 = Dot(color="#33FF57").scale(0.5)
        tangency2 = Dot(color="#33FF57").scale(0.5)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(cone))
        # Fix 20 & 22: Move sphere1 to D4-E5 area for better flow and no overlap
        self.place_in_area(sphere1, 'B4', 'C5', scale_factor=0.6) 
        self.play(FadeIn(sphere1))
        self.lecture[0].set_color(RED)

        # === Animation for Lecture Line 2 ===
        # Fix 21 & 22: Move sphere2 to F5 for consistent diagonal flow
        self.place_at_grid(sphere2, 'E5', scale_factor=0.5)
        self.play(FadeIn(sphere2))
        self.lecture[1].set_color(ORANGE)

        # === Animation for Lecture Line 3 ===
        # Place tangencies
        tangency1.move_to(sphere1.get_center())
        tangency2.move_to(sphere2.get_center())
        self.play(
            FadeIn(tangency1), 
            FadeIn(tangency2)
        )
        self.lecture[2].set_color("#33FF57")
        self.wait(2)
