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
        self.setup_layout("The Gradient: The Direction of Correction", [
            "The gradient is a compass for correction.",
            "It senses sensitivity to weight changes.",
            "We walk opposite the slope to descend."
        ])

        # === Animation for Lecture Line 1 ===
        axes = ThreeDAxes(x_range=[-2, 2], y_range=[-2, 2], z_range=[0, 2], x_length=4, y_length=4, z_length=2)
        surface = axes.plot_surface(
            lambda u, v: 0.5 * (u**2 + v**2),
            u_range=[-2, 2], v_range=[-2, 2],
            fill_opacity=0.6, color=BLUE
        )
        landscape = VGroup(axes, surface).rotate(PI/4, axis=RIGHT)
        self.place_in_area(landscape, 'D4', 'F6', scale_factor=0.5)
        
        # Label "Loss Landscape"
        label = Text("Loss Landscape", font_size=20, color=WHITE)
        self.place_at_grid(label, 'C4', scale_factor=0.7)
        
        # Asset: Compass
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        self.place_at_grid(compass, 'A5', scale_factor=0.6)
        
        self.play(FadeIn(landscape), Write(label), FadeIn(compass))
        self.lecture[0].set_color("#FFD700")

        # === Animation for Lecture Line 2 ===
        point = Dot(color="#FF6347")
        point_pos = axes.c2p(1.5, 1.5, 0.5 * (1.5**2 + 1.5**2))
        point.move_to(point_pos)
        
        grad = Vector(0.5 * RIGHT + 0.5 * UP, color="#FF6347")
        grad.move_to(point.get_center())
        
        self.add(point, grad)
        self.lecture[1].set_color("#FF6347")
        self.play(Create(point), GrowArrow(grad))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00CED1")
        target_pos = axes.c2p(0, 0, 0)
        self.play(
            point.animate.move_to(target_pos),
            grad.animate.move_to(target_pos)
        )
        self.wait(1)
