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

class Section4Scene(ThreeDScene, TeachingScene):
    def construct(self):
        self.setup_layout("Gradient Descent: Step-by-Step Adjustment", [
            "We visualize the gradient as the mountain's steepness.",
            "Gradient descent moves weights down the slope carefully.",
            "Each step brings us closer to minimum error."
        ])

        # === Animation for Lecture Line 1 ===
        # Load asset - mountain icon representation
        mountain = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mountain.svg")
        
        axes = ThreeDAxes(x_range=[-3, 3], y_range=[-3, 3], z_range=[0, 3])
        surface = axes.plot_surface(
            lambda x, y: 0.2 * (x**2 + y**2),
            u_range=[-2.5, 2.5],
            v_range=[-2.5, 2.5],
            color=WHITE
        )
        
        # Using mandated placement logic
        self.place_in_area(axes, 'B2', 'E5', scale_factor=0.7)
        self.place_in_area(surface, 'B2', 'E5', scale_factor=0.8)
        
        self.play(Create(axes), Create(surface))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Dot representing random weight state
        dot = Dot(color="#FF0000")
        self.place_at_grid(dot, 'C3', scale_factor=0.5)
        
        # Placing on the surface - need to align with axes
        dot.move_to(axes.c2p(2, 2, 0.8))
        self.add(dot)
        self.play(FadeIn(dot))
        self.lecture[1].set_color("#FF0000")

        # === Animation for Lecture Line 3 ===
        # Dot sliding down
        self.play(
            dot.animate.move_to(axes.c2p(0, 0, 0)),
            run_time=3,
            rate_func=linear
        )
        dot.set_color("#00FF00")
        self.lecture[2].set_color("#00FF00")
        self.wait(2)
