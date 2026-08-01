from manim import *
import numpy as np

# Base TeachingScene class as per requirements
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
        # Title and Lecture Lines from storyboard
        title_text = "Prerequisite: Functions as Mappings"
        lecture_lines = [
            "- Imagine two parallel lines for input and output.",
            "- Arrows show how input points map to output points.",
            "- This shifts focus from coordinates to relative lengths."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors defined in storyboard and planner instructions
        input_color = "#00FF00"
        output_color = "#00BFFF"
        dot_color = "#FFFF00"
        arrow_color = "#FFFFFF"
        
        # Define function for mapping (non-linear to show varying ratio)
        def f(x):
            return 0.2 * (x**2) + 0.5

        # === Animation for Lecture Line 1 ===
        # Imagine two parallel lines for input and output.
        input_line = NumberLine(
            x_range=[0, 5, 1], 
            length=4, 
            color=input_color, 
            include_numbers=True, 
            font_size=18,
            label_direction=UP
        )
        output_line = NumberLine(
            x_range=[0, 6, 1], 
            length=4, 
            color=output_color, 
            include_numbers=True, 
            font_size=18,
            label_direction=DOWN
        )
        
        # positioning from VideoCritic fixes (Issue 31)
        self.place_in_area(input_line, 'B3', 'B6')
        self.place_in_area(output_line, 'D3', 'D6')
        
        # Labels positioned within 1 grid unit (Issue 30)
        input_label = Text("Input (x)", font_size=18, color=input_color)
        output_label = Text("Output f(x)", font_size=18, color=output_color)
        
        self.place_at_grid(input_label, 'B2', scale_factor=0.8)
        self.place_at_grid(output_label, 'D2', scale_factor=0.8)

        # Highlight first lecture line
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(
            Create(input_line), 
            Create(output_line), 
            FadeIn(input_label), 
            FadeIn(output_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Arrows show how input points map to output points.
        mapping_points = [1, 2, 3, 4]
        arrows = VGroup()
        for x in mapping_points:
            start_p = input_line.n2p(x)
            end_p = output_line.n2p(f(x))
            # Create arrows with slight vertical offset for clarity
            arrow = Arrow(
                start_p + DOWN * 0.2, 
                end_p + UP * 0.2, 
                color=arrow_color, 
                stroke_width=2, 
                max_tip_length_to_length_ratio=0.1,
                buff=0
            )
            arrows.add(arrow)
            
        # Update lecture highlighting
        self.play(
            self.lecture[0].animate.set_color(WHITE), 
            self.lecture[1].animate.set_color(YELLOW)
        )
        self.play(LaggedStart(*[Create(a) for a in arrows], lag_ratio=0.3))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # This shifts focus from coordinates to relative lengths.
        input_start_val = 1
        input_tracker = ValueTracker(input_start_val)
        
        # Create segments to show relative length change (Storyboard 3)
        input_segment = Line(
            input_line.n2p(input_start_val), 
            input_line.n2p(input_start_val), 
            color=dot_color, 
            stroke_width=8
        )
        output_segment = Line(
            output_line.n2p(f(input_start_val)), 
            output_line.n2p(f(input_start_val)), 
            color=dot_color, 
            stroke_width=8
        )
        
        # Updaters for persistent mobjects (L010 preferred)
        input_segment.add_updater(lambda l: l.put_start_and_end_on(
            input_line.n2p(input_start_val), 
            input_line.n2p(input_tracker.get_value())
        ))
        output_segment.add_updater(lambda l: l.put_start_and_end_on(
            output_line.n2p(f(input_start_val)), 
            output_line.n2p(f(input_tracker.get_value()))
        ))
        
        # Dots at the moving ends for better visualization
        input_dot = Dot(color=dot_color, radius=0.1)
        output_dot = Dot(color=dot_color, radius=0.1)
        input_dot.add_updater(lambda d: d.move_to(input_line.n2p(input_tracker.get_value())))
        output_dot.add_updater(lambda d: d.move_to(output_line.n2p(f(input_tracker.get_value()))))

        # Dynamic focus label using Text for stability (L022)
        ratio_label = Text("Relating 'Input Move' to 'Output Move'", font_size=20, color=WHITE)
        # Fix from VideoCritic (Issue 32)
        self.place_in_area(ratio_label, 'E3', 'E6', scale_factor=0.9)
        
        # Update lecture highlighting
        self.play(
            self.lecture[1].animate.set_color(WHITE), 
            self.lecture[2].animate.set_color(YELLOW)
        )
        self.play(
            FadeIn(input_segment),
            FadeIn(output_segment),
            FadeIn(input_dot), 
            FadeIn(output_dot), 
            FadeIn(ratio_label),
            arrows.animate.set_stroke(opacity=0.2)
        )
        
        # Execute movement showing varying relative distance
        self.play(input_tracker.animate.set_value(4.5), run_time=6, rate_func=smooth)
        self.wait(2)
        
        # Final cleanup
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
