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
        self.setup_layout("The Concept of Iteration", [
            "Dynamic systems apply functions repeatedly.",
            "Orbits track a point's sequence of positions.",
            "Iteration turns simple rules into long paths."
        ])
        
        # Define assets (Using generic shape replacements since paths are abstract)
        planet = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/planet.svg")
        satellite = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/satellite.svg")
        
        circle = Circle(radius=1.0, color="#FF00FF")
        point = Dot(color=YELLOW)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF00FF")
        self.place_at_grid(circle, 'B4', scale_factor=1.0)
        self.place_at_grid(planet, 'B4', scale_factor=0.5)
        self.play(Create(circle), FadeIn(planet))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        self.place_at_grid(point, 'B4', scale_factor=0.6)
        self.play(FadeIn(point))
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFFF")
        
        path = VGroup()
        curr = complex(0.2, 0.2)
        
        # Animate satellite orbiting
        for i in range(3):
            next_val = curr**2
            
            # Map complex point to grid area (B4 to E6)
            new_dot = Dot(color=YELLOW, radius=0.05)
            # Simple scaling to keep it within B4-E6 bounds
            pos = np.array([4.5 + next_val.real * 0.5, 1.5 - next_val.imag * 0.5, 0])
            new_dot.move_to(pos)
            path.add(new_dot)
            
            self.play(
                point.animate.move_to(pos),
                satellite.animate.move_to(pos),
                FadeIn(new_dot)
            )
            curr = next_val
        
        self.play(FadeIn(satellite))
        self.wait(2)
