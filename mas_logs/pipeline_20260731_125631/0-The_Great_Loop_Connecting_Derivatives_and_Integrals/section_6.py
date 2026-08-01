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

class Section6Scene(TeachingScene):
    def construct(self):
        # Initializing the layout with storyboard data
        title_text = "Summary: The Calculus Circle"
        lecture_lines = [
            "Differentiation goes from position down to velocity.",
            "Integration goes from velocity back up to position.",
            "They form the great loop of calculus."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # A circular infographic appears with 'Position' at the top and 'Velocity' at the bottom.
        # [Lecture Line 1 color changes to #FFFF00]
        
        pos_text = Text("Position", font_size=24, color=WHITE)
        vel_text = Text("Velocity", font_size=24, color=WHITE)
        
        # Center labels horizontally within the loop's central columns
        self.place_in_area(pos_text, "B3", "B4", scale_factor=1.0)
        self.place_in_area(vel_text, "E3", "E4", scale_factor=1.0)
        
        self.play(
            Write(pos_text),
            Write(vel_text),
            self.lecture[0].animate.set_color("#FFFF00"),
            run_time=1.2
        )
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # A red (#FF0000) arrow sweeps from Position to Velocity, labeled 'Derivative'.
        # [Lecture Line 2 color changes to #FFFF00]
        
        # Derivative arrow on the right side of the diagram
        # Connecting row B (top) to row E (bottom) at column 5
        der_arrow = CurvedArrow(
            self.grid["B5"], 
            self.grid["E5"], 
            angle=-PI/2, 
            color="#FF0000"
        )
        der_label = Text("Derivative", font_size=18, color="#FF0000")
        # Place label near the arrow - Issue 32 fix
        self.place_in_area(der_label, "C5", "D5", scale_factor=0.8)
        
        self.play(
            Create(der_arrow),
            FadeIn(der_label),
            self.lecture[1].animate.set_color("#FFFF00"),
            run_time=1.2
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # A green (#00FF00) arrow sweeps from Velocity to Position, labeled 'Integral'.
        # [Lecture Line 3 color changes to #FFFF00]
        
        # Integral arrow on the left side of the diagram
        # Connecting row E (bottom) to row B (top) at column 2
        int_arrow = CurvedArrow(
            self.grid["E2"], 
            self.grid["B2"], 
            angle=-PI/2, 
            color="#00FF00"
        )
        int_label = Text("Integral", font_size=18, color="#00FF00")
        # Place label near the arrow - Issue 31 fix
        self.place_in_area(int_label, "C2", "D2", scale_factor=0.8)
        
        self.play(
            Create(int_arrow),
            FadeIn(int_label),
            self.lecture[2].animate.set_color("#FFFF00"),
            run_time=1.2
        )
        
        # Emphasize the infinite loop cycle
        self.play(
            der_arrow.animate.set_stroke(width=10),
            int_arrow.animate.set_stroke(width=10),
            rate_func=there_and_back,
            run_time=1.5
        )
        
        self.wait(2)
