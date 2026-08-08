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
        lecture_lines = [
            "Phase space visualizes collision states as trajectories.",
            "Reflections happen within a V-shaped mirror wedge.",
            "The trajectory angle depends on the mass ratio.",
            "Reflections count the slope to escape the wedge.",
            "Geometry interprets the physical collision count."
        ]
        self.setup_layout("Geometric Interpretation", lecture_lines)
        
        # Define colors for lines
        colors = ["#FFADAD", "#FFD6A5", "#FDFFB6", "#CAFFBF", "#9BF6FF"]

        # === Animation for Lecture Line 1 ===
        # Display a unit circle on canvas
        circle = Circle(radius=1.5, color=BLUE)
        self.place_in_area(circle, 'A2', 'C4', scale_factor=0.5)
        self.play(Create(circle))
        self.lecture[0].set_color(colors[0])

        # === Animation for Lecture Line 2 ===
        # Reflections happen within a V-shaped mirror wedge
        # Loading asset /scratch/pawsey1357/jthen/Code2Video/assets/icon/mirror.svg
        wedge_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mirror.svg")
        wedge = VGroup(
            Line(ORIGIN, 2 * DOWN + 2 * RIGHT),
            Line(ORIGIN, 2 * DOWN + 2 * LEFT),
            wedge_icon
        ).set_color(WHITE)
        self.place_in_area(wedge, 'D2', 'F4', scale_factor=0.6)
        self.play(FadeOut(circle), Create(wedge))
        self.lecture[1].set_color(colors[1])

        # === Animation for Lecture Line 3 ===
        # The trajectory angle depends on the mass ratio
        traj = Line(2 * DOWN + 1 * LEFT, 2 * UP + 1 * RIGHT, color=YELLOW)
        # Using asset mirror.svg again for trajectory reflection context
        reflect_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mirror.svg")
        self.place_in_area(reflect_icon, 'A4', 'B6', scale_factor=0.3)
        self.place_in_area(traj, 'B2', 'E5', scale_factor=0.8)
        self.play(Create(traj), FadeIn(reflect_icon))
        self.lecture[2].set_color(colors[2])

        # === Animation for Lecture Line 4 ===
        # Reflections count the slope to escape the wedge
        dot = Dot(color=RED)
        self.place_at_grid(dot, 'D5', scale_factor=0.8)
        self.add(dot)
        self.play(dot.animate.move_to(self.grid["B4"]))
        self.lecture[3].set_color(colors[3])

        # === Animation for Lecture Line 5 ===
        # Geometry interprets the physical collision count
        self.lecture[4].set_color(colors[4])
        self.wait(1)
