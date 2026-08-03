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
        
        # Colors
        input_color = "#ADD8E6"     # Light blue
        output_color = "#90EE90"    # Light green
        highlight_input_color = "#FFD700"  # Yellow
        highlight_output_color = "#FFA500" # Orange
        
        # Assets
        slider_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/slider.svg"
        
        # 1. Objects initialization
        # Input Space line
        input_line = NumberLine(
            x_range=[0, 4, 1],
            length=5,
            color=input_color,
            include_numbers=True,
            font_size=16,
            stroke_width=2
        )
        
        # Output Space line (f(x) = 2x)
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
        
        # Position objects using grid system
        self.place_in_area(input_line, "B1", "B6")
        self.place_in_area(output_line, "E1", "E6")
        
        # FIX for Issue 22 and 23: use place_in_area for labels to ensure centering and avoid overlap
        self.place_in_area(input_label, 'A1', 'A6', scale_factor=0.8)
        self.place_in_area(output_label, 'D1', 'D6', scale_factor=0.8)
        
        # Asset Loading (Persistent Mobjects)
        slider = SVGMobject(slider_path).set_height(0.3).set_color(WHITE)
        pointer = Dot(radius=0.1, color=WHITE)
        
        # ValueTracker for synchronization
        x_tracker = ValueTracker(1)
        
        # Updaters for real-time movement
        slider.add_updater(lambda m: m.move_to(input_line.n2p(x_tracker.get_value())))
        pointer.add_updater(lambda m: m.move_to(output_line.n2p(2 * x_tracker.get_value())))
        
        # Highlights to show displacement
        highlight_in = Line(color=highlight_input_color, stroke_width=4)
        highlight_in.add_updater(lambda m: m.put_start_and_end_on(
            input_line.n2p(1), 
            input_line.n2p(x_tracker.get_value())
        ))
        
        highlight_out = Line(color=highlight_output_color, stroke_width=4)
        highlight_out.add_updater(lambda m: m.put_start_and_end_on(
            output_line.n2p(2), 
            output_line.n2p(2 * x_tracker.get_value())
        ))

        # === Animation for Lecture Line 1 ===
        # Initial colors: highlight first, dim others
        for i in range(1, len(self.lecture)):
            self.lecture[i].set_color(GRAY)
            
        self.play(
            FadeIn(input_line),
            FadeIn(output_line),
            FadeIn(input_label),
            FadeIn(output_label),
            run_time=0.8
        )
        self.wait(0.5)
        
        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color(highlight_input_color),
            run_time=0.4
        )
        
        # Movement of input slider
        x_tracker.set_value(1)
        self.add(slider, highlight_in)
        self.play(x_tracker.animate.set_value(2), run_time=1.2, rate_func=linear)
        self.wait(0.5)
        
        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color(highlight_output_color),
            run_time=0.4
        )
        
        # Reset and synchronized movement for mapping
        x_tracker.set_value(1)
        self.add(pointer, highlight_out)
        self.play(x_tracker.animate.set_value(2), run_time=1.8, rate_func=smooth)
        self.wait(1.5)
