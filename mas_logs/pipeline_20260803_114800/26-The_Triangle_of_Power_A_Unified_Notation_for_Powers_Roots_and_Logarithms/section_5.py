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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup title and lecture lines
        title_str = "Operation 2: The Root (Right-to-Left View)"
        lines_str = [
            "- Finding a root means searching for the missing base.",
            "- We move from the result back to the start.",
            "- The triangle unifies radicals into this simple structure."
        ]
        self.setup_layout(title_str, lines_str)

        # Triangle vertices
        base_pos = self.grid["E2"]
        result_pos = self.grid["E6"]
        exponent_pos = self.grid["C4"] # Fixed from B4 per Issue 40 to avoid title crowding

        # Triangle lines
        side_left = Line(base_pos, exponent_pos, color=GRAY)
        side_right = Line(exponent_pos, result_pos, color=GRAY)
        side_bottom = Line(result_pos, base_pos, color=GRAY)
        triangle = VGroup(side_left, side_right, side_bottom)

        # Mobjects for values
        val_64 = Text("64", color="#FF0000", font_size=36)
        val_3 = Text("3", color="#00FF00", font_size=36)
        val_4 = Text("4", color="#0000FF", font_size=36)
        val_question = Text("?", color=WHITE, font_size=36)

        # Position values with updated scales per Issues 40, 41, 42
        self.place_at_grid(val_64, "E6", scale_factor=1.2)
        self.place_at_grid(val_3, "C4", scale_factor=1.2)
        self.place_at_grid(val_4, "E2", scale_factor=1.2)
        self.place_at_grid(val_question, "E2", scale_factor=1.2)

        # Labels for vertices
        lbl_res = Text("Result", font_size=18).next_to(val_64, DOWN, buff=0.2)
        lbl_exp = Text("Exponent", font_size=18).next_to(val_3, UP, buff=0.2)
        lbl_base = Text("Base", font_size=18).next_to(val_question, DOWN, buff=0.2)

        # === Animation for Lecture Line 1 ===
        # Finding a root means searching for the missing base.
        # Highlight in Blue as it's about the Base concept
        self.play(self.lecture[0].animate.set_color("#0000FF")) 
        self.play(
            Create(triangle),
            Write(val_64),
            Write(val_3),
            Write(val_question),
            Write(lbl_res),
            Write(lbl_exp),
            Write(lbl_base),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # We move from the result back to the start.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFFF00") # Yellow for the trace
        )

        # Yellow glowing lines tracing back
        trace_bottom = Line(result_pos, base_pos, color="#FFFF00", stroke_width=6)
        trace_top = Line(exponent_pos, base_pos, color="#FFFF00", stroke_width=6)
        
        self.play(
            Create(trace_bottom),
            Create(trace_top),
            run_time=2
        )
        self.play(
            FadeOut(trace_bottom),
            FadeOut(trace_top),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The triangle unifies radicals into this simple structure.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#0000FF") # Blue for revealed result
        )

        self.play(
            FadeOut(val_question),
            FadeIn(val_4),
            val_4.animate.scale(1.5),
            Flash(base_pos, color="#0000FF", line_length=0.4),
            run_time=1
        )
        self.play(val_4.animate.scale(1/1.5), run_time=0.5)
        
        self.wait(3)
