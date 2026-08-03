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
        title_text = "Operation 1: Solving for the Result (Powers)"
        lecture_lines = [
            "When we need the Result, we calculate powers.",
            "Given a Base of 3 and Exponent of 2.",
            "The missing bottom-right corner must be 9."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Define triangle using asset
        # Place triangle in the main area C3-E4, but scale it to fit nicely between labels
        triangle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/triangle.svg")
        triangle.set_color(WHITE)
        # Use place_in_area to center the triangle relative to labels
        # Labels will be at E2 (BL), B3-B4 (Top), E5 (BR)
        self.place_in_area(triangle, "B2", "E5", scale_factor=2.5)

        # Reference points for labels (consistent with Critic feedback)
        # Top vertex area
        top_pos = (self.grid["B3"] + self.grid["B4"]) / 2
        # Bottom Left vertex area
        bl_pos = self.grid["E2"]
        # Bottom Right vertex area
        br_pos = self.grid["E5"]

        # Define labels
        # Base at E2 (per Issue 29)
        base_3 = Text("3", font_size=36)
        base_3.move_to(bl_pos)
        
        # Exponent in area B3-B4 (per Issue 31)
        exp_2 = Text("2", font_size=36)
        self.place_in_area(exp_2, "B3", "B4", scale_factor=1.0)
        
        # Result at E5 (per Issue 30)
        res_q = Text("?", font_size=36, color="#FF4500")
        res_q.move_to(br_pos)
        
        # === Animation for Lecture Line 1 ===
        # Show the triangle [Asset] with 3 at bottom-left and 2 at the top; bottom-right is a '?' (#FF4500).
        self.play(self.lecture[0].animate.set_color("#FF4500"))
        
        self.play(DrawBorderThenFill(triangle))
        self.play(Write(base_3), Write(exp_2), Write(res_q))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Create a glowing yellow line (#FFFF00) connecting the Base (3) and Exponent (2).
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFFF00")
        )
        
        # Connecting line from Base (3) to Exponent (2)
        # Using grid-based positions to ensure it looks connected to the labels
        glow_line = Line(bl_pos, top_pos, color="#FFFF00", stroke_width=8)
        self.play(Create(glow_line))
        
        # Pulsing highlight effect
        self.play(glow_line.animate.set_stroke(width=14), run_time=0.4)
        self.play(glow_line.animate.set_stroke(width=8), run_time=0.4)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The '?' at the bottom-right transforms into the number 9 in lime green (#32CD32).
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#32CD32")
        )
        
        res_9 = Text("9", font_size=36, color="#32CD32")
        res_9.move_to(br_pos)
        
        # Transform '?' into '9'
        self.play(Transform(res_q, res_9))
        self.wait(2)
        
        # Cleanup: Return last lecture line to white
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)
