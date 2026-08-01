from manim import *
import numpy as np

# === Base Class ===
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

# === Section 1 ===
class Section1Scene(TeachingScene):
    def construct(self):
        # Fetch content from shared state
        title_text = "The Concept of an 'Infinite Loop'"
        lecture_lines = [
            "Iteration feeds a function's output back as input.",
            "This creates a sequence called an orbit.",
            "Points can explode to infinity or shrink away."
        ]
        
        # 1. Setup Layout
        self.setup_layout(title_text, lecture_lines)
        
        # Set initial lecture color to gray
        self.lecture.set_color(GRAY)
        
        # Colors (L008: Use hex)
        COLOR_FEEDBACK = "#FFFFFF"
        COLOR_ORBIT = "#00FF00"
        
        # Asset path
        DOT_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/dot.svg"
        
        # === Animation for Lecture Line 1 ===
        # Line: "Iteration feeds a function's output back as input."
        self.play(self.lecture[0].animate.set_color(COLOR_FEEDBACK))
        
        # Circular arrow loop
        loop_arc = Arc(radius=0.8, start_angle=0, angle=TAU * 0.8, color=COLOR_FEEDBACK)
        loop_arc.add_tip()
        loop_label = Text("Feedback Loop", font_size=20, color=COLOR_FEEDBACK)
        feedback_group = VGroup(loop_arc, loop_label).arrange(DOWN, buff=0.3)
        
        # Fix for Issue 21: Position feedback_group in A2-D5 area to avoid title crowding
        self.place_in_area(feedback_group, "A2", "D5", scale_factor=0.8)
        
        self.play(Create(loop_arc), Write(loop_label))
        self.wait(2)
        
        # === Animation for Lecture Line 2 ===
        # Line: "This creates a sequence called an orbit."
        self.play(self.lecture[1].animate.set_color(COLOR_ORBIT))
        
        # Number Line
        number_line = NumberLine(
            x_range=[0, 6, 1],
            length=5.0,
            include_numbers=True,
            include_tip=True,
            tip_width=0.15,
            tip_height=0.15,
            font_size=18,
            color=WHITE
        )
        # Fix for Issue 22: Move number_line to E2-F6 to avoid lecture text overlap
        self.place_in_area(number_line, "E2", "F6", scale_factor=0.9)
        
        # Pixel the Rabbit (using SVGMobject for Issue 19)
        # Load asset once
        dot = SVGMobject(DOT_ASSET)
        dot.set_color(COLOR_ORBIT)
        dot.scale(0.15) # Scale to dot size
        dot.move_to(number_line.n2p(2))
        
        # Label (L002: scaled and proximity)
        dot_label = Text("Pixel", font_size=16, color=COLOR_ORBIT)
        dot_label.next_to(dot, UP, buff=0.1)
        
        self.play(Create(number_line))
        self.play(FadeIn(dot), Write(dot_label))
        self.wait(1)
        
        # Jumps: 2 -> 4 -> 16 (off-screen)
        # Jump to 4
        self.play(
            dot.animate.move_to(number_line.n2p(4)),
            dot_label.animate.next_to(number_line.n2p(4), UP, buff=0.1),
            run_time=1.5
        )
        self.wait(0.5)
        
        # Jump to "infinity" (off-screen right)
        # number_line.n2p(12) is beyond the visible number line
        self.play(
            dot.animate.move_to(number_line.n2p(12)),
            dot_label.animate.move_to(number_line.n2p(12) + UP*0.3),
            run_time=1.5
        )
        self.play(FadeOut(dot), FadeOut(dot_label))
        self.wait(1.5)
        
        # === Animation for Lecture Line 3 ===
        # Line: "Points can explode to infinity or shrink away."
        self.play(self.lecture[2].animate.set_color(COLOR_ORBIT))
        
        # Reset Pixel at 0.5
        dot.move_to(number_line.n2p(0.5))
        dot_label.next_to(dot, UP, buff=0.1)
        
        self.play(FadeIn(dot), Write(dot_label))
        self.wait(0.5)
        
        # Shrink sequence: 0.5 -> 0.25 -> 0.0625 -> 0
        shrink_targets = [0.25, 0.0625, 0]
        for target in shrink_targets:
            self.play(
                dot.animate.move_to(number_line.n2p(target)),
                dot_label.animate.next_to(number_line.n2p(target), UP, buff=0.1),
                run_time=1.2
            )
        
        self.wait(2)
