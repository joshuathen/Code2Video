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
        # Initialize Scene
        title = "Grover's Step 1: The Oracle (Phase Inversion)"
        lines = [
            "The Oracle marks the target by flipping its sign.",
            "It reflects the target's amplitude across the axis.",
            "Only the target state becomes negative.",
            "This process is called phase inversion.",
            "It highlights the answer without changing its probability."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # The Oracle marks the target by flipping its sign.
        self.play(self.lecture[0].animate.set_color("#00BFFF"))
        
        # Issue 43: Align state labels using place_in_area
        state_labels = VGroup(*[Text(s, font_size=24, color=WHITE) for s in ["|00>", "|01>", "|10>", "|11>"]])
        state_labels.arrange(RIGHT, buff=0.4)
        self.place_in_area(state_labels, 'A2', 'A5', scale_factor=0.8)
        
        # Issue 44: Anchor amplitude bars using place_in_area
        # Height 3.0 ensures they reach the baseline at Row D when centered in B-C area
        bars = VGroup(*[
            Rectangle(
                height=3.0, 
                width=0.6, 
                fill_opacity=1.0, 
                fill_color="#00BFFF", 
                stroke_color="#00BFFF"
            ) for _ in range(4)
        ])
        bars.arrange(RIGHT, buff=0.2)
        self.place_in_area(bars, 'B2', 'C5', scale_factor=0.9)
        
        self.play(Create(bars), Write(state_labels))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # It reflects the target's amplitude across the axis.
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        
        # Issue 45: Anchor zero-axis line using place_in_area
        zero_axis = Line(LEFT*2.5, RIGHT*2.5, color="#FFFFFF", stroke_width=4)
        self.place_in_area(zero_axis, 'D1', 'D6', scale_factor=1.0)
        
        # Target bar is index 2 (|10>)
        target_outline = SurroundingRectangle(bars[2], color="#FFD700", buff=0.1, stroke_width=5)
        
        self.play(Create(zero_axis), Create(target_outline))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Only the target state becomes negative.
        self.play(self.lecture[2].animate.set_color("#FF6347"))
        
        # Phase Inversion: Symmetric reflection across the anchored axis
        # Use rotation about the axis center to achieve a perfect mirrored flip (Issue 55)
        self.play(
            bars[2].animate.rotate(PI, axis=RIGHT, about_point=zero_axis.get_center()).set_color("#FF6347"),
            target_outline.animate.rotate(PI, axis=RIGHT, about_point=zero_axis.get_center()),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # This process is called phase inversion.
        self.play(self.lecture[3].animate.set_color("#FF6347"))
        
        phase_text = Text("Phase Inversion", font_size=24, color=WHITE)
        self.place_at_grid(phase_text, "A6")
        
        self.play(Write(phase_text))
        self.play(Flash(bars[2], color="#FF6347", line_length=0.4, flash_radius=0.6))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # It highlights the answer without changing its probability.
        self.play(self.lecture[4].animate.set_color("#FFD700"))
        
        # Emphasize the highlighted state via outline pulse
        self.play(
            target_outline.animate.set_stroke(width=10),
            bars[2].animate.set_opacity(0.7),
            run_time=0.5
        )
        self.play(
            target_outline.animate.set_stroke(width=5),
            bars[2].animate.set_opacity(1.0),
            run_time=0.5
        )
        self.wait(2)
