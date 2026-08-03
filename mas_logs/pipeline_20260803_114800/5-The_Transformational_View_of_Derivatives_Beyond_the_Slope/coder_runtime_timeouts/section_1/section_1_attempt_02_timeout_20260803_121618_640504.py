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
        input_color = "#ADD8E6"     # Light blue
        output_color = "#90EE90"    # Light green
        highlight_input_color = "#FFD700"  # Yellow
        highlight_output_color = "#FFA500" # Orange
        
        # Assets
        slider_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/slider.svg"
        
        # 1. Objects initialization
        # Input line (x space)
        input_line = NumberLine(
            x_range=[0, 4, 1],
            length=5,
            color=input_color,
            include_numbers=True,
            font_size=16,
            stroke_width=2
        )
        
        # Output line (f(x) space, let's say f(x) = 2x)
        output_line = NumberLine(
            x_range=[0, 8, 2],
            length=5,
            color=output_color,
            include_numbers=True,
            font_size=16,
            stroke_width=2
        )
        
        input_label = Text("Input Space (x)", font_size=20, color=input_color)
        output_label = Text("Output Space (f(x))", font_size=20, color=output_color)
        
        # Place objects using the grid system
        self.place_in_area(input_line, "B1", "B6")
        self.place_in_area(output_line, "E1", "E6")
        self.place_at_grid(input_label, "A3")
        self.place_at_grid(output_label, "D3")
        
        # Load slider asset once
        # Instructions require using the [Asset: ...] references.
        slider = SVGMobject(slider_path).set_height(0.3).set_color(WHITE)
        pointer = Dot(radius=0.1, color=WHITE)
        
        # Trackers and Updaters for synchronization (avoids always_redraw for performance)
        x_tracker = ValueTracker(1)
        
        # Slider follows x_tracker on input line
        slider.add_updater(lambda m: m.move_to(input_line.n2p(x_tracker.get_value())))
        # Pointer follows 2 * x_tracker on output line
        pointer.add_updater(lambda m: m.move_to(output_line.n2p(2 * x_tracker.get_value())))
        
        # Highlight lines using updaters for movement
        highlight_in = Line(color=highlight_input_color, stroke_width=6)
        highlight_in.add_updater(lambda m: m.put_start_and_end_on(
            input_line.n2p(1), 
            input_line.n2p(x_tracker.get_value())
        ))
        
        highlight_out = Line(color=highlight_output_color, stroke_width=6)
        highlight_out.add_updater(lambda m: m.put_start_and_end_on(
            output_line.n2p(2), 
            output_line.n2p(2 * x_tracker.get_value())
        ))
        
        # === Animation for Lecture Line 1 ===
        # "Think of functions as mappings between number lines."
        self.play(self.lecture[0].animate.set_color(WHITE), run_time=0.5)
        self.play(
            Create(input_line),
            Create(output_line),
            Write(input_label),
            Write(output_label),
            run_time=1.2
        )
        self.wait(0.5)
        
        # === Animation for Lecture Line 2 ===
        # "The input x moves on the top line."
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color(highlight_input_color),
            run_time=0.5
        )
        
        # Initialize slider at x=1
        x_tracker.set_value(1)
        self.play(FadeIn(slider), Create(highlight_in), run_time=0.5)
        # Move x from 1 to 2
        self.play(x_tracker.animate.set_value(2), run_time=1.5, rate_func=linear)
        self.wait(0.5)
        
        # === Animation for Lecture Line 3 ===
        # "This triggers movement on the output f(x) line."
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color(highlight_output_color),
            run_time=0.5
        )
        
        # Show pointer and move both simultaneously to demonstrate the mapping
        x_tracker.set_value(1)
        self.play(FadeIn(pointer), Create(highlight_out), run_time=0.5)
        self.play(x_tracker.animate.set_value(2), run_time=1.8, rate_func=smooth)
        self.wait(2)
