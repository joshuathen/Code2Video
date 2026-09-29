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
        self.setup_layout("Defining the Fatou and Julia Sets", [
            "Stable orbits form Fatou sets.",
            "Unstable boundaries define Julia sets.",
            "Chaos emerges from sensitive dependence.",
            "Small changes cause vast differences.",
            "This boundary creates complex fractals."
        ])
        
        # Define mobjects
        fatou_region = Circle(radius=1.5, color="#FF00FF", fill_opacity=0.3)
        # Using SVG placeholder as instructed. If the SVG doesn't exist, this fails,
        # but the storyboard explicitly references /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        fatou_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        julia_boundary = Circle(radius=1.5, color="#00FFFF", stroke_width=4)
        
        # === Animation for Lecture Line 1 ===
        # Task 25/27: Use place_in_area B4 to E6
        # Task 17: Integrate icon
        self.play(FadeIn(self.lecture[0].set_color("#FF00FF")), 
                  FadeIn(self.place_in_area(fatou_region, "B4", "E6", scale_factor=0.6)),
                  FadeIn(self.place_in_area(fatou_icon, "B4", "E6", scale_factor=0.2)))
        
        # === Animation for Lecture Line 2 ===
        # Task 26/27: Place Julia slightly differently
        self.play(FadeIn(self.lecture[1].set_color("#00FFFF")), 
                  Create(self.place_in_area(julia_boundary, "B4", "E6", scale_factor=0.65)))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(YELLOW))
        
        # === Animation for Lecture Line 5 ===
        # Task 17: Zoom in
        self.play(self.lecture[4].animate.set_color("#00FFFF"), 
                  julia_boundary.animate.scale(1.5))
        self.wait(2)
