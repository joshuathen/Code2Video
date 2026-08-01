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

class Section3TheProcessScene(TeachingScene):
    def construct(self):
        # Initialize the layout with the title and lecture lines
        lecture_lines = [
            "First, differentiate both sides with respect to x.",
            "Second, collect all dy/dx terms on one side.",
            "Third, factor out and solve for dy/dx."
        ]
        self.setup_layout("The Three-Step Recipe", lecture_lines)

        # Define distinct colors for each step to link lecture lines with visual steps
        # Using light, distinguishable colors as per guidelines
        color_step1 = "#FFFF00"  # Yellow
        color_step2 = "#00FFFF"  # Cyan
        color_step3 = "#FFA500"  # Orange

        # === Animation for Lecture Line 1 ===
        # Visual: A rectangle outline (Recipe Card) appears containing the text for Step 1.
        # The rectangle spans the main right-side grid area.
        recipe_card = Rectangle(width=5.0, height=4.5, color=WHITE)
        self.place_in_area(recipe_card, "A1", "F6")
        
        step1_text = Text("Step 1: Differentiate Both Sides", font_size=24, color=color_step1)
        self.place_in_area(step1_text, "B1", "B6", scale_factor=0.9)
        
        self.play(
            self.lecture[0].animate.set_color(color_step1),
            Create(recipe_card),
            Write(step1_text),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Visual: Text for Step 2 appears inside the rectangle, below Step 1.
        step2_text = Text("Step 2: Collect dy/dx Terms", font_size=24, color=color_step2)
        self.place_in_area(step2_text, "C1", "C6", scale_factor=0.9)
        
        self.play(
            self.lecture[1].animate.set_color(color_step2),
            Write(step2_text),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Visual: Text for Step 3 appears inside the rectangle, below Step 2.
        step3_text = Text("Step 3: Factor and Isolate dy/dx", font_size=24, color=color_step3)
        self.place_in_area(step3_text, "D1", "D6", scale_factor=0.9)
        
        self.play(
            self.lecture[2].animate.set_color(color_step3),
            Write(step3_text),
            run_time=1.5
        )
        self.wait(2)
