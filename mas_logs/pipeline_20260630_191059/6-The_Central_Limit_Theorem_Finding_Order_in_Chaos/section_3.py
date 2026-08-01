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
        title = "The Magic: The Distribution of Sample Means"
        lines = [
            "Now we plot all those sample means together.",
            "Watch the shape of the new distribution emerge.",
            "Despite the messy population, a bell curve forms.",
            "This is the Central Limit Theorem in action.",
            "Randomness converges into a predictable normal shape."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # The sticky note stacks from Section 2 settle into a clear histogram. 
        # Label 'Distribution of Sample Means' in #FF4500.
        self.lecture[0].set_color(YELLOW)
        
        # Create a basic histogram using rectangles
        # Representing a distribution of means that is starting to center
        heights = [0.2, 0.5, 1.2, 1.8, 1.2, 0.5, 0.2]
        bars = VGroup(*[
            Rectangle(width=0.4, height=h, fill_opacity=0.6, fill_color=BLUE, stroke_width=1)
            for h in heights
        ]).arrange(RIGHT, buff=0.1)
        
        self.place_in_area(bars, "B2", "E5")
        
        hist_label = Text("Distribution of Sample Means", font_size=18, color="#FF4500")
        # Fix: Positioning hist_label in area A3-A4 for better centering (Issue 24)
        self.place_in_area(hist_label, "A3", "A4", scale_factor=0.8)
        
        self.play(Create(bars), Write(hist_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Display the original U-shaped population distribution as a ghosted outline in #555555 for comparison.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # U-shaped population (ghosted)
        u_shape = FunctionGraph(
            lambda x: 0.5 * (x**2),
            x_range=[-1.8, 1.8],
            color="#555555"
        )
        self.place_in_area(u_shape, "B2", "E5")
        
        self.play(Create(u_shape))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Animate the histogram bars shifting and growing as sample size 'n' increases from 2 to 30.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # New heights for a more concentrated distribution (larger n)
        new_heights = [0.05, 0.2, 1.5, 3.0, 1.5, 0.2, 0.05]
        new_bars = VGroup(*[
            Rectangle(width=0.3, height=h, fill_opacity=0.7, fill_color=BLUE_C, stroke_width=1)
            for h in new_heights
        ]).arrange(RIGHT, buff=0.05)
        self.place_in_area(new_bars, "B2", "E5")
        
        n_label = Text("n = 30", font_size=20, color=WHITE)
        # Fix: Moving n_label to B6 to avoid cluttering the plot area (Issue 26)
        self.place_at_grid(n_label, "B6", scale_factor=0.7)
        
        self.play(
            Transform(bars, new_bars),
            FadeIn(n_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Morph the histogram bars into a smooth, white (#FFFFFF) Bell Curve.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        bell_curve = FunctionGraph(
            lambda x: 3 * np.exp(-x**2 / 0.5),
            x_range=[-2, 2],
            color="#FFFFFF"
        )
        self.place_in_area(bell_curve, "B2", "E5")
        
        self.play(
            FadeOut(bars),
            FadeOut(n_label),
            Create(bell_curve)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Highlight the center of the Bell Curve and label 'Normal Distribution' in #00FA9A.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Defining a vertical center line to highlight the peak
        center_line = DashedLine(
            start=DOWN * 1.5,
            end=UP * 1.5,
            color="#00FA9A"
        )
        # Positioning center line at the peak of the bell curve (between col 3 and 4)
        self.place_in_area(center_line, "B3", "E4")
        
        norm_label = Text("Normal Distribution", font_size=20, color="#00FA9A")
        # Fix: Positioning norm_label in area F3-F4 for centering (Issue 25)
        self.place_in_area(norm_label, "F3", "F4", scale_factor=0.8)
        
        self.play(
            Create(center_line),
            Write(norm_label)
        )
        self.wait(2)
        
        self.lecture[4].set_color(WHITE)
        self.wait(1)
