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
        self.setup_layout("Prerequisite Review: Area as a Function", [
            "Area function accumulates area over time.", 
            "Sliding boundary defines the interval.", 
            "Integration is a dynamic area process."
        ])
        
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 8, 2], axis_config={"include_tip": False})
        curve = axes.plot(lambda t: 2*t, x_range=[0, 4], color=WHITE)
        
        # Initial area under curve
        x_val = ValueTracker(2.0)
        
        # Create area group
        area = always_redraw(lambda: axes.get_area(curve, x_range=[0, x_val.get_value()], color="#2ECC71", opacity=0.5))
        
        # Load asset
        boundary_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/boundary.svg")
        boundary_icon.set_color("#ECF0F1")
        
        # Use a dot as the marker and attach icon
        marker = Dot(color="#9B59B6")
        marker_group = VGroup(marker, boundary_icon)
        boundary_icon.next_to(marker, UP, buff=0.1)

        def update_marker(m):
            pos = axes.input_to_graph_point(x_val.get_value(), curve)
            m.move_to(pos)
            
        marker_group.add_updater(update_marker)
        
        # Positioning according to critical feedback
        self.place_in_area(axes, "B1", "D5", scale_factor=0.5)
        self.add(axes, curve, area, marker_group)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#2ECC71"))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#9B59B6"))
        self.play(x_val.animate.set_value(3.5), run_time=2)
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#ECF0F1"))
        self.play(x_val.animate.set_value(0.5), run_time=2)
        self.play(x_val.animate.set_value(3.8), run_time=2)
        self.wait(1)
