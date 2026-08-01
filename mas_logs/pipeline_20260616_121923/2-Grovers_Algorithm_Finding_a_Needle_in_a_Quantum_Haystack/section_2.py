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
        # Setup the layout with lecture lines
        lecture_lines = [
            'We begin by putting qubits into uniform superposition.', 
            'Every possible index now has an equal probability amplitude.', 
            'Visually, all state bars share the same initial height.'
        ]
        self.setup_layout("Prerequisite: Superposition and the State Space", lecture_lines)
        
        # Define colors for the animation
        WHITE_HEX = "#FFFFFF"
        BLUE_HEX = "#ADD8E6"
        GRAY_HEX = "#888888"

        # Dim lines initially to allow highlighting
        for line in self.lecture:
            line.set_color(GRAY_HEX)

        # === Animation for Lecture Line 1 ===
        # Highlight current line
        self.play(self.lecture[0].animate.set_color(WHITE_HEX))
        
        # Create 8 bars representing the 3-qubit state space
        bars = VGroup(*[
            Rectangle(
                height=1.0, 
                width=0.3, 
                fill_color=WHITE_HEX, 
                fill_opacity=1.0, 
                stroke_color=WHITE_HEX,
                stroke_width=1
            ) for _ in range(8)
        ]).arrange(RIGHT, buff=0.2)
        
        # Position bars in the lower area of the grid to avoid being top-heavy
        # Ref: Issue 30, 32
        self.place_in_area(bars, "C1", "E6", scale_factor=0.8)
        
        # Show bars appearing one by one
        self.play(LaggedStart(*[Create(bar) for bar in bars], lag_ratio=0.1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight current line with matching color
        self.play(self.lecture[1].animate.set_color(BLUE_HEX))
        
        # Create ket labels |000> through |111>
        # Replaced MathTex with Text to avoid environment issues
        labels = VGroup(*[
            Text(f"|{i:03b}>", font_size=18, color=BLUE_HEX) 
            for i in range(8)
        ]).arrange(RIGHT, buff=0.42) # Buff adjusted to align visually with the bars
        
        # Position labels at the base of the state space using the grid system
        # Ref: Issue 31
        self.place_in_area(labels, "F1", "F6", scale_factor=0.6)
            
        self.play(Write(labels))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight current line
        self.play(self.lecture[2].animate.set_color(BLUE_HEX))
        
        # Animate a soft light blue pulse across all bars to represent uniform superposition
        self.play(
            LaggedStart(
                *[Indicate(bar, color=BLUE_HEX, scale_factor=1.1) for bar in bars],
                lag_ratio=0.1
            ),
            run_time=2
        )
        self.wait(2)
