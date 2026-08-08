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
        # Fetching data from storyboard
        title_text = "Prerequisite: The Conservation Laws"
        lecture_lines = [
            "Physics is governed by energy and momentum conservation.",
            "Velocity states form an ellipse in our coordinate system.",
            "One axis represents mass m, the other mass M."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        COLOR_ENERGY = "#00FFFF"  # Cyan
        COLOR_MOMENTUM = "#FF00FF"  # Magenta
        COLOR_ELLIPSE = "#00FFFF"  # Cyan
        
        # === Animation for Lecture Line 1 ===
        # Physics is governed by energy and momentum conservation.
        
        energy_formula = MathTex(r"\frac{1}{2}mv^2 + \frac{1}{2}MV^2 = E", color=COLOR_ENERGY)
        momentum_formula = MathTex(r"mv + MV = p", color=COLOR_MOMENTUM)
        
        # Fix Issue 26: Energy formula too close to lecture text.
        self.place_in_area(energy_formula, 'C3', 'C4', scale_factor=0.7)
        
        # Fix Issue 27: Momentum formula positioning.
        self.place_in_area(momentum_formula, 'C5', 'C6', scale_factor=0.7)
        
        self.play(
            Write(energy_formula),
            Write(momentum_formula),
            self.lecture[0].animate.set_color(COLOR_ENERGY),
            run_time=2
        )
        
        # Pulse both formulas as per storyboard animation step 3
        self.play(
            energy_formula.animate.scale(1.2),
            momentum_formula.animate.scale(1.2),
            rate_func=there_and_back,
            run_time=1.5
        )
        
        # === Animation for Lecture Line 2 ===
        # Velocity states form an ellipse in our coordinate system.
        
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=3.5,
            y_length=2.5,
            axis_config={"include_tip": True, "color": WHITE}
        )
        
        # The ellipse representing the energy equation in phase space
        ellipse = Ellipse(
            width=2.8, 
            height=1.8, 
            color=COLOR_ELLIPSE,
            stroke_width=4
        )
        
        # Labels for the velocities (the actual axes) - Added to group for area placement
        label_v = MathTex("v", color=WHITE).scale(0.8)
        label_V = MathTex("V", color=WHITE).scale(0.8)
        label_v.next_to(axes.x_axis.get_end(), RIGHT, buff=0.1)
        label_V.next_to(axes.y_axis.get_end(), UP, buff=0.1)

        graph_group = VGroup(axes, ellipse, label_v, label_V)
        
        # Fix Issue 28: Graph group crowding lecture notes.
        self.place_in_area(graph_group, 'D2', 'F6', scale_factor=0.9)
        
        self.play(
            Create(axes),
            Create(ellipse),
            Write(label_v),
            Write(label_V),
            self.lecture[1].animate.set_color(COLOR_ELLIPSE),
            run_time=2
        )
        
        # === Animation for Lecture Line 3 ===
        # One axis represents mass m, the other mass M.
        
        # Conceptual mass labels as per script, placed relative to axes
        label_m = MathTex("m", color=COLOR_ENERGY).scale(0.7)
        label_M = MathTex("M", color=COLOR_ENERGY).scale(0.7)
        
        label_m.next_to(axes.x_axis, DOWN, buff=0.2)
        label_M.next_to(axes.y_axis, LEFT, buff=0.2)
        
        self.play(
            Write(label_m),
            Write(label_M),
            self.lecture[2].animate.set_color(COLOR_ENERGY),
            run_time=2
        )
        
        self.wait(3)
