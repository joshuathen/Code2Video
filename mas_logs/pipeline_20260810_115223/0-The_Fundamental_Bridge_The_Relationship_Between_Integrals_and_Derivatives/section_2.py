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
        self.setup_layout("The Conceptual Shift: Accumulation", [
            "Integration calculates the area under a curve.",
            "It accumulates tiny slices to form a total.",
            "Think of it as the sum of growth segments."
        ])
        
        # Define the curve and region
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 3, 1], axis_config={"include_tip": False}).scale(0.5)
        curve = axes.plot(lambda x: 0.2*x**2 + 0.5, color=WHITE)
        area = axes.get_area(curve, [0, 3.5], color="#3399FF", opacity=0.5)
        
        # Import Asset
        slice_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/slice.svg").scale(0.5)
        
        # Group for layout
        visual_group = VGroup(axes, curve, area, slice_icon)
        
        # Fixed placement based on recommendations
        self.place_in_area(visual_group, 'C2', 'F6', scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(visual_group))
        self.play(self.lecture_lines[0].animate.set_color("#3399FF"))

        # === Animation for Lecture Line 2 ===
        # Create growing slices
        slice_group = VGroup()
        for i in range(10):
            x_start = i * 0.35
            x_end = (i + 1) * 0.35
            rect = axes.get_riemann_rectangles(curve, x_range=[x_start, x_end], dx=0.35, color=YELLOW)
            slice_group.add(rect)
        
        self.play(FadeIn(slice_group), run_time=2)
        self.play(self.lecture_lines[1].animate.set_color(YELLOW))
        
        # === Animation for Lecture Line 3 ===
        self.play(Indicate(slice_group), run_time=2)
        self.play(self.lecture_lines[2].animate.set_color(ORANGE))
        
        self.wait(2)
