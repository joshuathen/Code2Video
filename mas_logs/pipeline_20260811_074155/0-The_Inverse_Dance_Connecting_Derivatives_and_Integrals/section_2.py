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
        # Setup layout with title and lecture lines
        # Fixed lecture lines to match storyboard + requested bullets
        self.setup_layout(
            "Prerequisite Bridge: Slopes and Areas", 
            [
                "- A derivative is the slope of a tangent line.", 
                "- An integral is the area trapped under a curve.", 
                "- These geometric properties are two sides of one coin."
            ]
        )
        
        # Colors defined in animation description and script
        BLUE_CURVE = "#1E90FF"
        RED_TANGENT = "#FF0000"
        YELLOW_RECTS = "#FFFF00"
        CONNECTION_COLOR = "#FFD700" # Gold

        # === Animation for Lecture Line 1 ===
        # Script: "A derivative is the slope of a tangent line."
        self.lecture[0].set_color(RED_TANGENT)
        
        # Define Axes on the grid - Addressing Issue #26: shift to right to avoid obstruction
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 4, 1],
            x_length=5,
            y_length=4,
            axis_config={"include_tip": True}
        )
        # Fix: Using 'A2' to 'F6' to prevent tangent line from overlapping lecture notes
        self.place_in_area(axes, 'A2', 'F6', scale_factor=0.8)
        
        # f(x) = 0.5 * (x - 2)^2 + 1
        func = lambda x: 0.5 * (x - 2)**2 + 1
        curve = axes.plot(func, x_range=[0, 4], color=BLUE_CURVE)
        
        # Addressing Issue #27: Move curve label to B6 to avoid overlap with shifted axes
        curve_label = MathTex("f(x)", color=BLUE_CURVE, font_size=24)
        self.place_at_grid(curve_label, 'B6', scale_factor=0.8)

        # Tangent Logic using ValueTracker and Updaters for efficiency
        t = ValueTracker(0.5)
        
        # Create persistent tangent line and dot
        tangent_line = TangentLine(curve, alpha=t.get_value() / 4.0, length=2.5, color=RED_TANGENT)
        dot = Dot(axes.c2p(t.get_value(), func(t.get_value())), color=RED_TANGENT)
        
        # Define Updaters to update mobjects in-place without heavy redraws
        def update_tangent(mob):
            new_line = TangentLine(curve, alpha=t.get_value() / 4.0, length=2.5, color=RED_TANGENT)
            mob.become(new_line)
            
        def update_dot(mob):
            mob.move_to(axes.c2p(t.get_value(), func(t.get_value())))

        tangent_line.add_updater(update_tangent)
        dot.add_updater(update_dot)

        # Initial drawing
        self.play(Create(axes), Create(curve), Write(curve_label), run_time=1.5)
        self.play(Create(tangent_line), Create(dot))
        
        # Animate tangent sliding along the curve
        self.play(t.animate.set_value(3.5), run_time=3, rate_func=there_and_back)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Script: "An integral is the area trapped under a curve."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW_RECTS)
        
        # Create Riemann Rectangles to represent the integral
        rects = axes.get_riemann_rectangles(
            curve, 
            x_range=[0.5, 3.5], 
            dx=0.3, 
            fill_opacity=0.6, 
            stroke_width=0.1, 
            color=YELLOW_RECTS
        )
        
        self.play(Create(rects), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Script: "These geometric properties are two sides of one coin."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(CONNECTION_COLOR)
        
        # Move the tangent line and rectangles together to emphasize connection
        # Return tangent to a central position and highlight area
        self.play(
            t.animate.set_value(2),
            rects.animate.set_fill(opacity=0.8),
            run_time=2
        )
        self.wait(3)
