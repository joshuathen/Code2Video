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
        lecture_lines = [
            "Backpropagation is assigning blame for errors.",
            "Traverse backward from output to the input layers.",
            "Each weight is adjusted by its contribution to error.",
            "Managers blame leads, leads blame individual developers.",
            "Everyone learns from their specific mistakes."
        ]
        self.setup_layout("The Logic of Backpropagation", lecture_lines)
        
        # Define mobjects
        manager = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/manager.svg")
        developer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/developer.svg")
        
        # Create a simple visual representing the system
        network = VGroup(manager, developer).arrange(RIGHT, buff=2)
        
        loss_curve = Axes(x_range=[-2, 2], y_range=[0, 4], axis_config={"include_tip": False}).scale(0.4)
        curve = loss_curve.plot(lambda x: x**2, color=WHITE)
        loss_curve_group = VGroup(loss_curve, curve)
        
        # Final grid container as per requirements
        grid_group = VGroup(network, loss_curve_group).arrange(DOWN)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        self.play(FadeIn(manager))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFD700")
        # Animate backward pulse (simplified)
        pulse = Dot(color="#FFD700").move_to(manager.get_left())
        self.play(pulse.animate.move_to(developer.get_right()))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFD700")
        # Fix: Using placement constraint from feedback
        self.place_in_area(loss_curve_group, 'C3', 'F5', scale_factor=0.6)
        self.play(Create(loss_curve_group))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFD700")
        gradient_tex = MathTex(r"Gradient = \partial Error / \partial Weight", color="#FFD700")
        # Fix: Using placement constraint from feedback
        self.place_at_grid(gradient_tex, 'D5', scale_factor=0.7)
        self.play(Write(gradient_tex))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFD700")
        # Fix: Using placement constraint from feedback
        self.place_in_area(grid_group, 'B2', 'F6', scale_factor=0.9)
        self.play(FadeIn(developer))
        self.wait(1)
