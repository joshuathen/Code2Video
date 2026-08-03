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
        # Define the lecture lines based on the storyboard
        lecture_lines = [
            "Think of functions as mappings between number lines.",
            "The input x moves on the top line.",
            "This triggers movement on the output f(x) line."
        ]
        
        # Initialize layout
        self.setup_layout("Prerequisite: The Mapping Concept", lecture_lines)
        
        # Colors from storyboard/constraints
        input_color = "#ADD8E6"
        output_color = "#90EE90"
        highlight_input_color = "#FFD700"
        highlight_output_color = "#FFA500"
        
        # Objects
        # Input line (x space)
        input_line = NumberLine(
            x_range=[0, 5, 1],
            length=5,
            color=input_color,
            include_numbers=True,
            font_size=18,
            stroke_width=2
        )
        
        # Output line (f(x) space)
        output_line = NumberLine(
            x_range=[0, 10, 2],
            length=5,
            color=output_color,
            include_numbers=True,
            font_size=18,
            stroke_width=2
        )
        
        input_label = Text("Input Space (x)", font_size=20, color=input_color)
        output_label = Text("Output Space (f(x))", font_size=20, color=output_color)
        
        # Place objects using the grid system
        self.place_in_area(input_line, 'B1', 'B6')
        self.place_in_area(output_line, 'E1', 'E6')
        self.place_at_grid(input_label, 'A3')
        self.place_at_grid(output_label, 'D3')
        
        # === Animation for Lecture Line 1 ===
        # "Think of functions as mappings between number lines."
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(
            Create(input_line),
            Create(output_line),
            Write(input_label),
            Write(output_label)
        )
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # "The input x moves on the top line."
        # Transition lecture highlight
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color(highlight_input_color)
        )
        
        # Setup slider and highlight tracker
        # ValueTracker tracks x
        x_tracker = ValueTracker(1)
        
        # Slider is a circle on the input line
        slider = Circle(radius=0.1, color=WHITE, fill_opacity=1, stroke_width=1)
        slider.move_to(input_line.n2p(1))
        
        # Add updater to slider to follow x_tracker
        slider.add_updater(lambda m: m.move_to(input_line.n2p(x_tracker.get_value())))
        
        # Highlight for input movement (yellow line segment)
        highlight_in = Line(input_line.n2p(1), input_line.n2p(1), color=highlight_input_color, stroke_width=6)
        highlight_in.add_updater(lambda m: m.put_start_and_end_on(input_line.n2p(1), input_line.n2p(x_tracker.get_value())))
        
        self.play(FadeIn(slider), Create(highlight_in))
        
        # Move slider from x=1 to x=2
        self.play(x_tracker.animate.set_value(2), run_time=2, rate_func=linear)
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        # "This triggers movement on the output f(x) line."
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color(highlight_output_color)
        )
        
        # Pointer for output space (f(x) = 2x)
        pointer = Circle(radius=0.1, color=WHITE, fill_opacity=1, stroke_width=1)
        pointer.move_to(output_line.n2p(2 * 1)) # Start at f(1)=2
        
        # Highlight for output movement (orange line segment)
        highlight_out = Line(output_line.n2p(2), output_line.n2p(2), color=highlight_output_color, stroke_width=6)
        
        # Updaters for pointer and output highlight
        pointer.add_updater(lambda m: m.move_to(output_line.n2p(2 * x_tracker.get_value())))
        highlight_out.add_updater(lambda m: m.put_start_and_end_on(output_line.n2p(2), output_line.n2p(2 * x_tracker.get_value())))
        
        # To show "Simultaneously", we reset and move both
        # First fade in the output components at their starting position (f(1)=2)
        # Note: x_tracker is currently at 2, so we need to set it back to 1 temporarily to initialize
        x_tracker.set_value(1)
        self.play(
            FadeIn(pointer),
            Create(highlight_out)
        )
        
        # Now move both simultaneously from x=1 to x=2
        self.play(x_tracker.animate.set_value(2), run_time=2.5, rate_func=smooth)
        self.wait(2)
