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
        # Mandatory call to setup_layout with snapshot lecture lines
        # This initializes the title and the lecture bullets on the left
        self.setup_layout(
            "The Principle of Least Time (Fermat's Principle)", 
            [
                "Fermat's Principle states light takes the path of least time.", 
                "Total time is the sum of travel times per medium.", 
                "We express time using distances and velocities for each path."
            ]
        )

        # Define formulas
        # Animation for Lecture Line 2: Display the total time formula
        formula1 = Text("T = (Path 1 / v₁) + (Path 2 / v₂)", font_size=24, color=WHITE)
        self.place_in_area(formula1, "B1", "B6", scale_factor=1.0)

        # Animation for Lecture Line 3: Substitute lengths
        # Using pink color #FF00FF as requested by the animation description
        formula2 = Text(
            "T(x) = [√(h₁² + x²) / v₁] + [√(h₂² + (L-x)²) / v₂]", 
            font_size=20, 
            color="#FF00FF"
        )
        self.place_in_area(formula2, "D1", "D6", scale_factor=1.0)

        # Snell's Law goal/result (Resolving Issue 40)
        # Fix: Centered horizontally using area C1 to C6
        snells_law = Text("n₁ sin(θ₁) = n₂ sin(θ₂)", font_size=24, color=YELLOW)
        self.place_in_area(snells_law, 'C1', 'C6', scale_factor=1.0)

        # === Animation for Lecture Line 1 ===
        # "Fermat's Principle states light takes the path of least time."
        self.play(self.lecture[0].animate.set_color(YELLOW), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Total time is the sum of travel times per medium."
        # Formula 1 is White, highlight line in Yellow for visibility
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW),
            Write(formula1),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "We express time using distances and velocities for each path."
        # Highlight matches the Pink color of formula 2
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FF00FF"),
            Write(formula2),
            run_time=1.5
        )
        self.wait(1)

        # === Additional: Resulting Goal (Related to Issue 40) ===
        # Showing the objective of the upcoming derivation
        self.play(
            Write(snells_law),
            run_time=1.5
        )
        self.play(
            snells_law.animate.scale(1.1),
            rate_func=there_and_back,
            run_time=1.5
        )
        self.wait(3)
