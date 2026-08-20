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
        self.setup_layout("The Formula and Application", [
            "Coordinates relate via [v]_B = P[v]_C.", 
            "Inverse matrices reverse the coordinate conversion.", 
            "This powers systems like self-driving car navigation."
        ])
        
        # === Animation for Lecture Line 1 ===
        formula = MathTex(r"[v]_B = P [v]_C", font_size=40)
        # Using place_in_area as requested in issue 32/26
        self.place_in_area(formula, "B4", "B6", scale_factor=0.9)
        self.play(Write(formula))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        inv_formula = MathTex(r"[v]_C = P^{-1} [v]_B", font_size=40, color="#FF8C00")
        car_icon = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png")
        self.place_in_area(inv_formula, "C4", "C6", scale_factor=0.9)
        self.place_at_grid(car_icon, "C2", scale_factor=0.5)
        self.play(FadeIn(car_icon), Write(inv_formula))
        self.lecture[1].set_color("#FF8C00")

        # === Animation for Lecture Line 3 ===
        gps_label = Text("Navigation System", font_size=24, color="#32CD32")
        # Shift down by 0.5, using D2 (now D2+0.5down) as requested in 27/32
        self.place_at_grid(gps_label, "D4", scale_factor=0.7)
        self.play(FadeIn(gps_label))
        self.lecture[2].set_color("#32CD32")
        self.wait(2)
