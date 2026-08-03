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
        # Setup layout with title and lecture lines
        lecture_lines = [
            "A positive derivative preserves the input's direction.",
            "A negative derivative flips the number line's orientation.",
            "A zero derivative collapses local space into one point."
        ]
        self.setup_layout("The Meaning of Sign and Zero", lecture_lines)

        # Shared assets: Number lines
        # Position input line across B2-B5 for better centering
        input_line = NumberLine(x_range=[-2, 2, 1], length=4, include_tip=True, color=GRAY)
        output_line = NumberLine(x_range=[-2, 2, 1], length=4, include_tip=True, color=GRAY)
        
        self.place_in_area(input_line, 'B2', 'B5')
        self.place_in_area(output_line, 'E2', 'E5')
        
        input_label = Text("Input", font_size=18, color=GRAY)
        output_label = Text("Output", font_size=18, color=GRAY)
        
        # Fix: Positioning labels using area for better centering (Resolving Issues 27 and 28)
        self.place_in_area(input_label, 'A2', 'A5', scale_factor=1.0)
        self.place_in_area(output_label, 'D2', 'D5', scale_factor=1.0)

        self.add(input_line, output_line, input_label, output_label)

        # === Animation for Lecture Line 1 ===
        # A positive derivative preserves the input's direction.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Arrow on input line moving right
        input_arrow_1 = Arrow(start=LEFT*0.3, end=RIGHT*0.3, color=YELLOW, buff=0, stroke_width=4)
        output_arrow_1 = Arrow(start=LEFT*0.3, end=RIGHT*0.3, color=YELLOW, buff=0, stroke_width=4)
        
        # Initial positions
        input_arrow_1.move_to(input_line.number_to_point(-1))
        output_arrow_1.move_to(output_line.number_to_point(-1))
        
        self.play(FadeIn(input_arrow_1), FadeIn(output_arrow_1))
        
        # Move them to the right (positive derivative)
        self.play(
            input_arrow_1.animate.move_to(input_line.number_to_point(1)),
            output_arrow_1.animate.move_to(output_line.number_to_point(1)),
            run_time=2
        )
        self.wait(1)
        self.play(FadeOut(input_arrow_1), FadeOut(output_arrow_1))
        self.play(self.lecture[0].animate.set_color(WHITE))

        # === Animation for Lecture Line 2 ===
        # A negative derivative flips the number line's orientation.
        self.play(self.lecture[1].animate.set_color(RED))
        
        input_arrow_2 = Arrow(start=LEFT*0.3, end=RIGHT*0.3, color=RED, buff=0, stroke_width=4)
        # Flip orientation: points LEFT (this represents the 'flip')
        output_arrow_2 = Arrow(start=RIGHT*0.3, end=LEFT*0.3, color=RED, buff=0, stroke_width=4)
        
        input_arrow_2.move_to(input_line.number_to_point(-1))
        # For a "flip" (e.g. f(x) = -x), as input moves right, output moves left
        output_arrow_2.move_to(output_line.number_to_point(1))
        
        self.play(FadeIn(input_arrow_2), FadeIn(output_arrow_2))
        
        # Animate movement: input moves right, output moves left
        self.play(
            input_arrow_2.animate.move_to(input_line.number_to_point(1)),
            output_arrow_2.animate.move_to(output_line.number_to_point(-1)),
            run_time=2
        )
        self.wait(1)
        self.play(FadeOut(input_arrow_2), FadeOut(output_arrow_2))
        self.play(self.lecture[1].animate.set_color(WHITE))

        # === Animation for Lecture Line 3 ===
        # A zero derivative collapses local space into one point.
        self.play(self.lecture[2].animate.set_color(BLUE))
        
        # Input segment on Input Line
        input_segment = Line(
            input_line.number_to_point(-0.5), 
            input_line.number_to_point(0.5), 
            color=BLUE, 
            stroke_width=6
        )
        # Output point (dot) on Output Line
        output_dot = Dot(output_line.number_to_point(0), color=BLUE, radius=0.08)
        
        self.play(FadeIn(input_segment))
        self.wait(0.5)
        
        # Animate "collapsing" the segment to the dot on the output line
        self.play(
            ReplacementTransform(input_segment.copy(), output_dot),
            run_time=2
        )
        
        self.wait(2)
        self.play(FadeOut(input_segment), FadeOut(output_dot))
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)
