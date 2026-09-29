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
        lecture_lines = [
            "ODEs describe change over time.",
            "Algebraic equations solve for fixed points.",
            "ODEs map dynamic trajectories."
        ]
        self.setup_layout("The Hook: Dynamics in Motion", lecture_lines)
        
        # Setup Animation Objects
        curve = ParametricFunction(
            lambda t: np.array([t, 0.5 * np.sin(t * 3), 0]),
            t_range=[-1.5, 1.5]
        ).set_color("#00FF00")
        
        # Load asset
        projectile = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/projectile.svg")
        projectile.set_color("#00FF00")
        
        velocity_vector = Arrow(ORIGIN, RIGHT * 0.5, color="#FF0000", buff=0)
        
        # Attach vector to projectile
        velocity_vector.add_updater(lambda m: m.next_to(projectile, RIGHT, buff=0))
        
        # Add visual area
        path_group = VGroup(curve, projectile, velocity_vector)
        self.place_in_area(path_group, 'C2', 'E5', scale_factor=1.2)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        self.play(Create(curve), FadeIn(projectile))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        fixed_point = Dot(color=WHITE)
        self.place_at_grid(fixed_point, 'E3', scale_factor=1.0)
        self.play(FadeIn(fixed_point))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.play(FadeIn(velocity_vector))
        self.play(MoveAlongPath(projectile, curve), run_time=3)
        self.wait(1)
