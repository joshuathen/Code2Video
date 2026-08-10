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
        self.setup_layout("Visualizing the Unfolding", [
            "Represent velocities as vectors in 2D.",
            "The wedge angle depends on mass ratio.",
            "Light beams hit edges π/θ times.",
            "The system measures Pi geometrically."
        ])
        
        # Assets
        mirror_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/mirror.svg"
        mirror = SVGMobject(mirror_path)
        
        # === Animation for Lecture Line 1 ===
        # Represent velocities as vectors in 2D.
        self.lecture[0].set_color("#FFFF00")
        wedge = VGroup(
            Line(ORIGIN, 3 * RIGHT + 1 * UP, color="#FFA500"),
            Line(ORIGIN, 3 * RIGHT + 1 * DOWN, color="#FFA500"),
            mirror.copy().scale(0.5).rotate(PI/4)
        )
        # Applying requested layout fix
        self.place_in_area(wedge, 'B3', 'D5', scale_factor=0.4)
        self.play(Create(wedge))

        # === Animation for Lecture Line 2 ===
        # The wedge angle depends on mass ratio.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Light beams hit edges π/θ times.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFFF00")
        dot = Dot(color=WHITE)
        # Applying requested layout fix
        self.place_at_grid(dot, 'C4', scale_factor=0.6)
        self.add(dot)
        
        path = VMobject(color="#D3D3D3")
        path.set_points_as_corners([self.grid['C4'], self.grid['D4'], self.grid['B5']])
        self.play(Create(path), MoveAlongPath(dot, path))

        # === Animation for Lecture Line 4 ===
        # The system measures Pi geometrically.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FF00FF")
        
        count_text = Text("Bounces: 3", color="#FF00FF", font_size=24)
        # Applying requested layout fix
        self.place_at_grid(count_text, 'E5', scale_factor=0.9)
        
        glow = mirror.copy().set_color("#FF00FF")
        self.play(Write(count_text), FadeIn(glow))
        self.wait(2)
