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
        self.setup_layout("Hook: The Universal Constant", [
            "Circles are everywhere in our universe.", 
            "[Asset: spinning_bicycle_wheel]", 
            "From small wheels to massive planets.", 
            "[Asset: planet_orbiting_star]", 
            "They all share a secret ratio."
        ])
        
        # Define Assets
        bicycle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bicycle.svg")
        star = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/star.svg")
        
        # === Animation for Lecture Line 1 ===
        # Circles are everywhere in our universe.
        self.lecture[0].set_color(BLUE)
        circle = Circle(radius=1.5, color=WHITE)
        pi_label = Text("Pi", color=GOLD, font_size=36)
        self.place_in_area(circle, 'B3', 'E5', scale_factor=0.6)
        self.place_at_grid(pi_label, 'C5', scale_factor=0.9)
        self.play(FadeIn(circle), Write(pi_label))

        # === Animation for Lecture Line 2 ===
        # [Asset: spinning_bicycle_wheel]
        self.lecture[1].set_color(BLUE)
        self.place_at_grid(bicycle, "C4", scale_factor=0.5)
        self.play(FadeIn(bicycle))
        self.play(bicycle.animate.set_stroke(color=YELLOW, width=4))

        # === Animation for Lecture Line 3 ===
        # From small wheels to massive planets.
        self.lecture[2].set_color(BLUE)
        dot_c = Dot(color=RED)
        dot_d = Dot(color=GREEN)
        
        # Position dots on circle perimeter and center
        dot_c.move_to(circle.get_right())
        dot_d.move_to(circle.get_center())
        
        self.place_at_grid(dot_c, 'B6', scale_factor=0.7)
        self.place_at_grid(dot_d, 'D4', scale_factor=0.7)
        
        dot_c_label = Text("C", font_size=20, color=RED).next_to(dot_c, UP)
        dot_d_label = Text("D", font_size=20, color=GREEN).next_to(dot_d, DOWN)
        
        self.play(FadeIn(star), FadeOut(bicycle)) # Fade in star while fading out bicycle
        self.play(FadeIn(dot_c), Write(dot_c_label), FadeIn(dot_d), Write(dot_d_label))

        # === Animation for Lecture Line 4 ===
        # [Asset: planet_orbiting_star]
        self.lecture[3].set_color(BLUE)
        self.place_at_grid(star, "B4", scale_factor=0.5)
        radius_line = Line(dot_d.get_center(), dot_c.get_center(), color=YELLOW)
        r_label = Text("r", font_size=20, color=YELLOW).next_to(radius_line, UP)
        self.play(Create(radius_line), Write(r_label))

        # === Animation for Lecture Line 5 ===
        # They all share a secret ratio.
        self.lecture[4].set_color(BLUE)
        secret_text = Text("Shared Secret", color=YELLOW, font_size=30)
        self.place_at_grid(secret_text, "E4")
        self.play(Write(secret_text))
        self.play(Flash(pi_label, color=YELLOW, line_length=0.2, num_lines=15))
        self.wait(2)
