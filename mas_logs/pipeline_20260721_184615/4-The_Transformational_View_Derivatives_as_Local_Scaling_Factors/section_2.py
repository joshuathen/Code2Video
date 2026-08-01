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
        # Data from storyboard/outline
        title = "Prerequisite: Linear Scaling and Uniform Stretch"
        lines = [
            "Linear functions stretch or squish the entire space uniformly.",
            "Constant factor c scales every interval by a fixed amount.",
            "Here, the scaling factor is a constant three everywhere."
        ]
        self.setup_layout(title, lines)
        
        # Colors (Hex only)
        COLOR_INPUT = "#87CEEB"
        COLOR_OUTPUT = "#FF6347"
        COLOR_SCALE = "#ADFF2F"
        COLOR_TEXT = "#FFFFFF"
        
        # === Animation for Lecture Line 1 ===
        # Highlight lecture line
        self.play(self.lecture[0].animate.set_color(COLOR_INPUT))
        
        # Input line setup
        # Using length such that 1 unit = 0.27 units
        input_line = NumberLine(
            x_range=[-3, 3, 1], 
            length=1.66, 
            include_numbers=True, 
            font_size=16,
            color=COLOR_TEXT
        )
        self.place_in_area(input_line, 'B2', 'B6')
        
        input_label = Text("Input Space", font_size=20, color=COLOR_TEXT)
        self.place_in_area(input_label, 'A2', 'A6', scale_factor=0.8)
        
        # Segment of length L [0, 1]
        segment_l = Line(
            start=input_line.n2p(0), 
            end=input_line.n2p(1), 
            color=COLOR_INPUT, 
            stroke_width=8
        )
        label_l = Text("L", color=COLOR_INPUT, font_size=20)
        self.place_at_grid(label_l, 'A5', scale_factor=0.8)
        
        self.play(Create(input_line), Write(input_label))
        self.play(Create(segment_l), FadeIn(label_l))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line
        self.play(self.lecture[1].animate.set_color(COLOR_OUTPUT))
        
        # Output line setup
        # Using length such that 3 units = 0.83 units (approx 3x larger than 0.27)
        output_line = NumberLine(
            x_range=[-9, 9, 3], 
            length=5.0, 
            include_numbers=True, 
            font_size=16,
            color=COLOR_TEXT
        )
        # Issue 23: Fix: Move output_line to Row F (F1-F6)
        self.place_in_area(output_line, 'F1', 'F6')
        
        # Issue 22: Fix: Move output_label to Row E (E1-E6)
        output_label = Text("Output Space (f(x)=3x)", font_size=20, color=COLOR_TEXT)
        self.place_in_area(output_label, 'E1', 'E6', scale_factor=0.8)
        
        # Scaled Segment 3L [0, 3]
        segment_3l = Line(
            start=output_line.n2p(0), 
            end=output_line.n2p(3), 
            color=COLOR_OUTPUT, 
            stroke_width=8
        )
        label_3l = Text("3L", color=COLOR_OUTPUT, font_size=20)
        self.place_at_grid(label_3l, 'E5', scale_factor=0.8)
        
        # Scaling Label and Arrow
        scale_text = Text("Scale = 3", color=COLOR_SCALE, font_size=22)
        self.place_in_area(scale_text, 'C2', 'C5', scale_factor=0.8)
        
        # Issue 21: Fix: Move arrow to Row D (D2-D5)
        arrow = Arrow(
            start=LEFT*0.7, 
            end=RIGHT*0.7, 
            color=COLOR_SCALE,
            buff=0.1
        )
        self.place_in_area(arrow, 'D2', 'D5')

        self.play(Create(output_line), Write(output_label))
        self.play(
            ReplacementTransform(segment_l.copy(), segment_3l),
            FadeIn(label_3l)
        )
        self.play(Write(scale_text), Create(arrow))
        self.play(Indicate(scale_text))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line
        self.play(self.lecture[2].animate.set_color(COLOR_SCALE))
        
        # Additional segments to show uniform scaling
        # Segment 1: [1, 2] -> [3, 6]
        seg_in_1 = Line(input_line.n2p(1), input_line.n2p(2), color=COLOR_INPUT, stroke_width=4)
        seg_out_1 = Line(output_line.n2p(3), output_line.n2p(6), color=COLOR_OUTPUT, stroke_width=4)
        
        # Segment 2: [-2, -1] -> [-6, -3]
        seg_in_2 = Line(input_line.n2p(-2), input_line.n2p(-1), color=COLOR_INPUT, stroke_width=4)
        seg_out_2 = Line(output_line.n2p(-6), output_line.n2p(-3), color=COLOR_OUTPUT, stroke_width=4)
        
        self.play(Create(seg_in_1), Create(seg_in_2))
        self.play(
            ReplacementTransform(seg_in_1.copy(), seg_out_1),
            ReplacementTransform(seg_in_2.copy(), seg_out_2)
        )
        
        self.wait(2.0)
