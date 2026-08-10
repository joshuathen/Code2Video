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
        self.setup_layout("Visualizing Probability Amplitudes", [
            "Amplitudes squared give us probability densities.",
            "Waves interfere, reinforcing or canceling outcomes.",
            "A robot explores every maze path simultaneously."
        ])

        # Assets
        robot_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        maze_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/maze.svg")

        # === Animation for Lecture Line 1 ===
        # Represent probability amplitudes as complex numbers on the plane.
        # Draw a unit circle
        circle = Circle(radius=1, color=WHITE)
        self.place_in_area(circle, 'C1', 'E3', scale_factor=0.55)
        self.place_at_grid(robot_icon, 'F2', scale_factor=0.5)
        
        self.play(Create(circle), FadeIn(robot_icon))
        self.lecture[0].set_color("#FF6666")

        # === Animation for Lecture Line 2 ===
        # Fade in phasor arrows
        phasor = Vector(direction=RIGHT, color="#FFFF00")
        phasor.move_to(circle.get_center())
        self.play(FadeIn(phasor))
        
        # Sector for probability modulus squared
        sector = Sector(radius=0.9, angle=PI/4, color="#FFFF00", fill_opacity=0.3)
        sector.move_to(circle.get_center())
        self.add(sector)
        
        # Rotate phasor and update sector
        self.play(Rotate(phasor, angle=PI, about_point=circle.get_center(), run_time=2))
        self.lecture[1].set_color("#FFFF66")

        # === Animation for Lecture Line 3 ===
        # Robot explores maze
        self.place_in_area(maze_icon, 'B4', 'E6', scale_factor=0.4)
        self.play(FadeIn(maze_icon))
        self.play(robot_icon.animate.move_to(maze_icon.get_center()), run_time=1)
        self.play(robot_icon.animate.set_opacity(0.3), run_time=1)
        self.lecture[2].set_color("#66FF66")
