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
        self.setup_layout("Prerequisite: Angular Sorting", ["The order depends on vector angles.", "Use cross-products for relative orientation.", "Avoid complex trigonometric calculations."])
        
        # Assets
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(protractor, "B2", scale_factor=0.5)
        self.play(FadeIn(protractor))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))

        # === Animation for Lecture Line 2 ===
        cross_prod_label = Text("Cross Product > 0", font_size=20, color=GREEN)
        self.place_at_grid(cross_prod_label, "A4", scale_factor=0.7)
        self.play(Write(cross_prod_label))
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        
        # === Animation for Lecture Line 3 ===
        no_trig_label = Text("No Trig Functions!", font_size=20, color=ORANGE)
        self.place_at_grid(no_trig_label, "E4", scale_factor=0.7)
        self.place_at_grid(compass, "D4", scale_factor=0.5)
        compass.set_color("#00FF00")
        self.play(Write(no_trig_label), FadeIn(compass))
        self.play(self.lecture[2].animate.set_color("#FFA500"))
        
        self.wait(2)
