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

class Section3Scene(TeachingScene):
    def construct(self):
        # Define the title and lecture lines for Section 3
        title_text = "Step 1: The Oracle (Phase Inversion)"
        lecture_lines = [
            "We use an Oracle to identify the correct answer.",
            "The Oracle recognizes the target state without measuring.",
            "It flips the sign of the target's probability amplitude.",
            "This process is known as phase inversion.",
            "Other states remain completely unchanged by the Oracle."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        TARGET_COLOR = "#FFD700"
        FLIPPED_COLOR = "#FF4500"
        NORMAL_COLOR = WHITE
        
        # === Animation for Lecture Line 1 ===
        # Show a collection of white vertical bars representing quantum states.
        self.lecture[0].set_color(NORMAL_COLOR)
        # Using 6 bars ensures the 4th bar aligns perfectly with grid column 4
        bars = VGroup(*[
            Rectangle(width=0.4, height=1.0, color=NORMAL_COLOR, fill_opacity=0.8, stroke_width=2) 
            for _ in range(6)
        ]).arrange(RIGHT, buff=0.5)
        
        # Fix: Place bars in C1-D6 area so their bottom is at the center of Row D (-0.8)
        self.place_in_area(bars, 'C1', 'D6')
        
        self.play(Create(bars), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight one specific bar (the 4th) with the label 'Target' in #FFD700.
        self.lecture[1].set_color(TARGET_COLOR)
        target_bar = bars[3] # 4th bar
        target_label = Text("Target", font_size=20, color=TARGET_COLOR)
        # Fix: Position target_label at B4 with scale 0.8 to associate with 4th bar
        self.place_at_grid(target_label, 'B4', scale_factor=0.8)
        
        self.play(
            target_bar.animate.set_color(TARGET_COLOR),
            Write(target_label),
            run_time=1.0
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Slowly flip the target bar across the X-axis so it points downwards.
        self.lecture[2].set_color(TARGET_COLOR)
        
        # Baseline is at the center of Row D (-0.8)
        base_y = -0.8
        flip_point = np.array([target_bar.get_center()[0], base_y, 0])
        
        self.play(
            target_bar.animate.scale(np.array([1, -1, 1]), about_point=flip_point),
            run_time=2.0
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Change the color of the flipped bar to #FF4500 and show a dashed line at Y=0.
        self.lecture[3].set_color(FLIPPED_COLOR)
        
        # Create a dashed line representing the zero-amplitude axis at the base of the bars (Row D).
        dashed_line = DashedLine(
            start=self.grid['D1'] + LEFT * 0.5,
            end=self.grid['D6'] + RIGHT * 0.5,
            color=WHITE,
            stroke_width=2
        )
        # Using place_in_area for D1-D6 centers the line exactly at y=-0.8
        line_container = VGroup(dashed_line)
        self.place_in_area(line_container, 'D1', 'D6')
        
        self.play(
            target_bar.animate.set_color(FLIPPED_COLOR),
            Create(dashed_line),
            run_time=1.0
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Pulse the other white bars to show they are unaffected.
        self.lecture[4].set_color(WHITE)
        
        others = VGroup(*[bars[i] for i in range(len(bars)) if i != 3])
        self.play(
            others.animate.set_color(WHITE).scale(1.1),
            rate_func=there_and_back,
            run_time=1.5
        )
        self.wait(2)
