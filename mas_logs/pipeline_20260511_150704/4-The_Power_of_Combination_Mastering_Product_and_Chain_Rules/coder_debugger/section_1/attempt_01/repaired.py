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
        # Initial Setup
        lines = [
            'A derivative represents the instantaneous rate of change.',
            'Calc-Bot’s velocity is the slope of the tangent line.',
            'But how do we differentiate products or nested functions?'
        ]
        self.setup_layout("Foundation: Derivative as a Rate of Change", lines)
        
        # Colors from description
        CALCBOT_COLOR = "#00FF00"
        CURVE_COLOR = "#1E90FF"
        TANGENT_COLOR = "#FFFF00"
        TEXT_WHITE = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Highlight current point in lecture
        self.lecture[0].set_color(CURVE_COLOR)
        
        # Define the math coordinate system
        axes = Axes(
            x_range=[-0.5, 2.5, 1],
            y_range=[0, 6, 1],
            axis_config={"include_tip": True},
            x_length=4.5,
            y_length=4.5
        )
        self.place_in_area(axes, "B1", "F6")
        
        curve = axes.plot(lambda x: x**2, x_range=[0, 2.3], color=CURVE_COLOR)
        # Using Text instead of MathTex to avoid FileNotFoundError: 'latex'
        curve_label = Text("y = x²", color=CURVE_COLOR, font_size=24)
        self.place_at_grid(curve_label, "A3")

        # Create Calc-Bot (the character)
        calc_bot = Dot(color=CALCBOT_COLOR, radius=0.12)
        bot_text = Text("Calc-Bot", color=CALCBOT_COLOR, font_size=20)
        
        # Track position
        x_tracker = ValueTracker(0.1)
        
        # Update bot position along curve
        calc_bot.add_updater(lambda m: m.move_to(axes.c2p(x_tracker.get_value(), x_tracker.get_value()**2)))
        bot_text.add_updater(lambda m: m.next_to(calc_bot, UP, buff=0.15))

        self.play(Create(axes), Create(curve), Write(curve_label))
        self.add(calc_bot, bot_text)
        
        # Visualizing Calc-Bot moving along the curve
        self.play(x_tracker.animate.set_value(1.8), run_time=4, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Highlight next lecture point
        self.lecture[1].set_color(TANGENT_COLOR)
        
        # Tangent line that tracks with the bot
        tangent_line = Line(color=TANGENT_COLOR, stroke_width=4)
        
        def update_tangent(line):
            val = x_tracker.get_value()
            p_current = axes.c2p(val, val**2)
            # Find direction vector in screen coordinates using a small offset
            eps = 0.001
            p_next = axes.c2p(val + eps, (val + eps)**2)
            direction = p_next - p_current
            unit_dir = direction / np.linalg.norm(direction)
            
            # Extend line by 1.5 units in each direction
            line.set_points_as_corners([
                p_current - unit_dir * 1.5,
                p_current + unit_dir * 1.5
            ])

        tangent_line.add_updater(update_tangent)
        
        self.play(Create(tangent_line))
        # Move back and forth to show slope changing
        self.play(x_tracker.animate.set_value(0.5), run_time=2)
        self.play(x_tracker.animate.set_value(2.0), run_time=2)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Highlight the final line
        self.lecture[2].set_color(TEXT_WHITE)
        
        # Derivative/Slope label - Using Text instead of MathTex to avoid FileNotFoundError: 'latex'
        derivative_label = Text("dy/dx = slope", color=TEXT_WHITE, font_size=24)
        self.place_at_grid(derivative_label, "B5")
        
        self.play(Write(derivative_label))
        # Final movement to tie everything together
        self.play(x_tracker.animate.set_value(1.2), run_time=3)
        self.wait(2)