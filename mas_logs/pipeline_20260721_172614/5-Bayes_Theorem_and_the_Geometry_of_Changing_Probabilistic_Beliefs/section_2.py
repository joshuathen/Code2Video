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
        # Title and Lecture Lines
        title_text = "Prerequisite: Probability as Area"
        lecture_lines = [
            "Imagine the entire universe of possibilities as a square.",
            "The square's total area equals a probability of one.",
            "Every specific event is a smaller rectangle inside it."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        GOLD = "#FFD700"
        GRAY = "#A9A9A9"
        WHITE_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Line 1: Imagine the entire universe of possibilities as a square.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Draw a white square #FFFFFF with side length 1, labeled 'Total Area = 1'.
        # We use a 3x3 area for visibility.
        main_square = Rectangle(width=3, height=3, color=WHITE_COLOR, stroke_width=2)
        self.place_in_area(main_square, "B2", "E5")
        
        self.play(Create(main_square), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2: The square's total area equals a probability of one.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        total_label = Text("Total Area = 1", font_size=24, color=WHITE_COLOR)
        self.place_in_area(total_label, "A2", "A5")
        
        self.play(Write(total_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3: Every specific event is a smaller rectangle inside it.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Vertical slice (width 0.2) in #FFD700 (Gold), labeled 'Bone in Roses (20%)'.
        # Remaining area (width 0.8) in #A9A9A9 (Dark Gray) as 'Elsewhere (80%)'.
        
        gold_slice = Rectangle(
            width=0.6, height=3, 
            fill_color=GOLD, fill_opacity=0.6, 
            stroke_color=GOLD, stroke_width=2
        )
        gray_slice = Rectangle(
            width=2.4, height=3, 
            fill_color=GRAY, fill_opacity=0.3, 
            stroke_color=GRAY, stroke_width=1
        )
        
        slices = VGroup(gold_slice, gray_slice).arrange(RIGHT, buff=0)
        self.place_in_area(slices, "B2", "E5")
        
        gold_label = Text("Bone in Roses (20%)", font_size=16, color=GOLD)
        gray_label = Text("Elsewhere (80%)", font_size=16, color=GRAY)
        
        # Position labels within 1 grid unit of slices
        # Adjusted positions and scaling per VideoCritic issues 31 and 32
        self.place_in_area(gold_label, 'F1', 'F3', scale_factor=0.7)
        self.place_in_area(gray_label, 'F4', 'F6', scale_factor=0.7)
        
        self.play(
            FadeIn(gold_slice),
            FadeIn(gray_slice),
            Write(gold_label),
            Write(gray_label)
        )
        self.wait(3)
        
        # Cleanup color
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
