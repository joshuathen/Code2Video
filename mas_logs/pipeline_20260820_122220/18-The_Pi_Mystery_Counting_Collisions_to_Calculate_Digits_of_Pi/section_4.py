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
        self.setup_layout("Visualizing the Physics: Phase Space Mapping", [
            "Map velocities as 2D coordinates.",
            "Each bounce reflects off a circle.",
            "The number of bounces equals Pi."
        ])
        
        # Load Assets
        billiard_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/billiard.svg")
        
        # Define objects
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"include_tip": True}).scale(0.6)
        label_x = MathTex("v_1", color=WHITE).scale(0.7).next_to(axes.x_axis, RIGHT)
        label_y = MathTex("v_2", color=WHITE).scale(0.7).next_to(axes.y_axis, UP)
        
        phase_space = VGroup(axes, label_x, label_y, billiard_icon.copy().scale(0.3))
        
        # Geometry for reflections
        arc = Arc(radius=1.5, angle=PI/2, start_angle=0, color=YELLOW).rotate(PI/4)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.place_in_area(phase_space, "B3", "E6", scale_factor=0.6)
        self.play(Create(phase_space))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        self.place_at_grid(arc, "D4", scale_factor=0.5)
        self.play(Create(arc))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFFF")
        dot = Dot(color="#00FFFF").move_to(arc.get_start())
        billiard_final = billiard_icon.copy()
        self.place_at_grid(billiard_final, "D4", scale_factor=0.3)
        self.play(Create(dot), FadeIn(billiard_final))
        self.play(dot.animate.move_to(arc.get_end()), run_time=2)
        self.wait(1)
