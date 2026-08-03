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
        # Data from storyboard
        title_text = "Visualizing Accumulation: The Area Under the Curve"
        lecture_lines = [
            "Visually, the integral represents the area under a curve.",
            "It sums up infinite tiny changes over an interval.",
            "This total area equals the accumulated value of change."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        CURVE_COLOR = "#FF69B4"
        LABEL_COLOR = "#FFFF00"
        SHADE_COLOR = "#FF69B4"
        
        # Setup Axes and Graph Elements
        axes = Axes(
            x_range=[0, 6, 1],
            y_range=[0, 4, 1],
            x_length=4.5,
            y_length=3.0,
            axis_config={"color": WHITE, "include_tip": True},
        )
        # Fix Issue 38: Move axes to 'B2' to 'E6' to avoid crowding with lecture text
        self.place_in_area(axes, "B2", "E6")
        
        func = lambda x: 0.1 * (x - 1) * (x - 3) * (x - 5) + 2
        curve = axes.plot(func, x_range=[0.5, 5.5], color=CURVE_COLOR)
        
        f_label = MathTex("f(x)", color=CURVE_COLOR).scale(0.8)
        # Fix Issue 39: Move f_label to 'A5' to avoid overlap with curve
        self.place_at_grid(f_label, "A5")
        
        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line in curve color
        self.play(self.lecture[0].animate.set_color(CURVE_COLOR))
        
        # Create axes and curve
        self.play(Create(axes), run_time=1)
        self.play(Create(curve), Write(f_label), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second lecture line in curve color (accumulation)
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(CURVE_COLOR)
        )
        
        # ValueTracker for the sweep animation
        x_tracker = ValueTracker(0.5)
        
        # Shaded area that updates with the tracker
        # Re-using get_area inside always_redraw for smooth accumulation
        shade = always_redraw(lambda: axes.get_area(
            curve, 
            x_range=[0.5, x_tracker.get_value()], 
            color=SHADE_COLOR, 
            opacity=0.3
        ))
        
        # Vertical sweeping line
        v_line = always_redraw(lambda: axes.get_vertical_line(
            axes.input_to_graph_point(x_tracker.get_value(), curve),
            color=WHITE
        ))
        
        self.add(shade, v_line)
        self.play(x_tracker.animate.set_value(5.5), run_time=4, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third lecture line in label color (result)
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(LABEL_COLOR)
        )
        
        # Final label for the integral/accumulation
        integral_label = Text("Integral", font_size=24, color=LABEL_COLOR)
        # Fix Issue 40: Move integral_label to 'D4' and scale to 1.0 for centering
        self.place_at_grid(integral_label, "D4", scale_factor=1.0)
        
        self.play(Write(integral_label))
        self.wait(3)
        
        # Final cleanup
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
