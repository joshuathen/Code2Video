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
        self.setup_layout("The Dynamic Area Function", [
            "Let's define a function that tracks accumulated area.",
            "A(x) starts at a fixed point a.",
            "It ends at a moving variable point x.",
            "As x moves right, the area shield grows.",
            "This creates a new function representing total area."
        ])

        # Colors
        CURVE_COLOR = "#FFFFFF"
        AREA_COLOR = "#00FFFF"
        TEXT_COLOR = "#00FFFF"
        SHIELD_COLOR = "#FFFFFF"

        # Coordinates/Axes for the graph
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 4, 1],
            x_length=4.5,
            y_length=3.5,
            axis_config={"include_tip": True, "color": WHITE}
        )
        self.place_in_area(axes, "B2", "F6")

        # Function curve: f(t) = 0.1 * t^2 + 1
        curve = axes.plot(lambda t: 0.1 * t**2 + 1, x_range=[0, 4.5], color=CURVE_COLOR)
        
        # Labels for axis
        # Issue 29: Use place_at_grid for labels instead of manual positioning
        f_label = MathTex("f(t)", font_size=24, color=WHITE)
        self.place_at_grid(f_label, "B2", scale_factor=0.8)
        
        t_label = MathTex("t", font_size=24, color=WHITE)
        self.place_at_grid(t_label, "F6", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Create(axes), Write(t_label), Write(f_label), run_time=1)
        self.play(Create(curve), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Point 'a' fixed at x=1
        a_val = 1.0
        a_line = axes.get_vertical_line(axes.c2p(a_val, curve.underlying_function(a_val)), color=WHITE)
        a_label = MathTex("a", font_size=24, color=WHITE).next_to(a_line, DOWN, buff=0.1)
        
        # Issue 24: Asset integration (shield icon at 'a')
        shield = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/shield.svg")
        shield.set_color(SHIELD_COLOR)
        shield.scale(0.2)
        shield.move_to(axes.c2p(a_val, 0.3))
        
        self.play(Create(a_line), Write(a_label), FadeIn(shield))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Tracker for moving 'x'
        x_tracker = ValueTracker(1.5)
        
        # Vertical line for 'x'
        x_line = Line(
            start=axes.c2p(1.5, 0),
            end=axes.c2p(1.5, curve.underlying_function(1.5)),
            color=WHITE
        )
        # Use updater for dynamic movement
        x_line.add_updater(lambda m: m.put_start_and_end_on(
            axes.c2p(x_tracker.get_value(), 0),
            axes.c2p(x_tracker.get_value(), curve.underlying_function(x_tracker.get_value()))
        ))
        
        # Label for 'x'
        x_label = MathTex("x", font_size=24, color=WHITE)
        x_label.add_updater(lambda m: m.next_to(x_line, DOWN, buff=0.1))
        
        self.play(Create(x_line), Write(x_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Area defined between 'a' and 'x'
        # always_redraw for area polygon
        area = always_redraw(lambda: axes.get_area(
            curve, 
            x_range=[a_val, x_tracker.get_value()], 
            color=AREA_COLOR, 
            opacity=0.5
        ))
        
        self.add(area)
        self.play(x_tracker.animate.set_value(4.0), run_time=3, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Issue 30: Label A(x) at B5
        ax_label = MathTex("A(x)", color=TEXT_COLOR, font_size=32)
        self.place_at_grid(ax_label, "B5", scale_factor=0.9)
        
        # Arrow pointing to the moving area
        arrow = Arrow(
            start=ax_label.get_bottom(),
            end=axes.c2p(2.5, 0.5),
            color=TEXT_COLOR,
            buff=0.1
        )
        # Update arrow to point to the center of the current area
        arrow.add_updater(lambda m: m.put_start_and_end_on(
            ax_label.get_bottom(),
            axes.c2p((a_val + x_tracker.get_value()) / 2, 0.5)
        ))
        
        self.play(Write(ax_label), GrowArrow(arrow))
        
        # Oscillate x to show A(x) dependency
        self.play(x_tracker.animate.set_value(3.0), run_time=1)
        self.play(x_tracker.animate.set_value(4.2), run_time=1)
        
        self.wait(2)
        self.lecture[4].set_color(WHITE)
