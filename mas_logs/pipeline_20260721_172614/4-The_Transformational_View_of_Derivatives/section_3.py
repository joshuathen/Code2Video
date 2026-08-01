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
        # Data
        title = "The 'Zoom' Effect: Local Linearity"
        lecture_lines = [
            "Globally, most functions appear curved and complex.",
            "However, zooming in at a point reveals local linearity.",
            "At this tiny scale, the function looks like a line.",
            "The derivative represents the scaling factor at this spot.",
            "It tells us how local space is being transformed."
        ]
        
        self.setup_layout(title, lecture_lines)
        
        # Colors
        COLOR_CURVE = "#FFFFFF"
        COLOR_ANT = "#FF0000"
        COLOR_DX = "#00FFFF"
        COLOR_DY = "#FF00FF"
        COLOR_FORMULA = "#FFFF00"
        
        # Grid center for zoomed alignment
        grid_center = (self.grid['A1'] + self.grid['F6']) / 2
        
        # === Animation for Lecture Line 1 ===
        # "Globally, most functions appear curved and complex."
        self.lecture[0].set_color(COLOR_CURVE)
        
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 10, 2],
            x_length=5,
            y_length=5,
            axis_config={"color": GREY_B, "stroke_width": 2},
            tips=False
        )
        curve = axes.plot(lambda x: x**2, x_range=[0, 3.2], color=COLOR_CURVE)
        plot_group = VGroup(axes, curve)
        self.place_in_area(plot_group, 'A1', 'F6', scale_factor=0.8)
        
        self.play(Create(axes), Create(curve))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # "However, zooming in at a point reveals local linearity."
        self.lecture[1].set_color(COLOR_ANT)
        
        # Point at (2,4) with Micro-Ant asset
        ant = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ant.svg")
        ant.set_color(COLOR_ANT)
        ant.set_height(0.3)
        ant.move_to(axes.c2p(2, 4))
        
        ant_label = Text("Point (2,4)", font_size=16, color=COLOR_ANT)
        # Fix for Issue 27: self.place_in_area(ant_label, 'A4', 'B5', scale_factor=0.8)
        self.place_in_area(ant_label, 'A4', 'B5', scale_factor=0.8)
        
        self.play(FadeIn(ant), Write(ant_label))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        # "At this tiny scale, the function looks like a line."
        self.lecture[2].set_color(WHITE)
        
        zoom_factor = 10
        zoom_point = ant.get_center().copy()
        
        # Scaling the view to reveal local linearity
        self.play(
            FadeOut(ant_label),
            plot_group.animate.scale(zoom_factor, about_point=zoom_point).shift(grid_center - zoom_point),
            ant.animate.move_to(grid_center),
            run_time=3
        )
        self.wait(1)
        
        # === Animation for Lecture Line 4 ===
        # "The derivative represents the scaling factor at this spot."
        self.lecture[3].set_color(COLOR_DX)
        
        # Slope at x=2 is 4.
        dx_val = 0.6
        dy_val = 2.4 # Slope 4
        
        dx_arrow = Arrow(
            grid_center, 
            grid_center + RIGHT * dx_val, 
            color=COLOR_DX, 
            buff=0,
            stroke_width=6
        )
        dy_arrow = Arrow(
            grid_center + RIGHT * dx_val, 
            grid_center + RIGHT * dx_val + UP * dy_val, 
            color=COLOR_DY, 
            buff=0,
            stroke_width=6
        )
        
        dx_text = MathTex("dx", color=COLOR_DX, font_size=24)
        dy_text = MathTex("dy", color=COLOR_DY, font_size=24)
        
        # Using grid system for label positioning
        self.place_at_grid(dx_text, 'E4', scale_factor=1.0)
        # Fix for Issue 26: self.place_at_grid(dy_text, 'F3', scale_factor=1.0)
        self.place_at_grid(dy_text, 'F3', scale_factor=1.0)
        
        self.play(GrowArrow(dx_arrow), Write(dx_text))
        self.play(GrowArrow(dy_arrow), Write(dy_text))
        self.wait(1)
        
        # === Animation for Lecture Line 5 ===
        # "It tells us how local space is being transformed."
        self.lecture[4].set_color(COLOR_FORMULA)
        
        formula = MathTex("dy = 4 \cdot dx", color=COLOR_FORMULA, font_size=32)
        # Fix for Issue 25: self.place_in_area(formula, 'A3', 'B5', scale_factor=0.9)
        self.place_in_area(formula, 'A3', 'B5', scale_factor=0.9)
        
        self.play(Write(formula))
        self.wait(2)
