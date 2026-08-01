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
        self.setup_layout("Prerequisite: Superposition and State Representation", [
            "Quantum computers search all possibilities simultaneously.",
            "We start by putting all states into superposition.",
            "Every state now has the same probability amplitude."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Draw a bar graph with 8 white (#FFFFFF) vertical bars of equal height, labeled |000> through |111> at the base.
        
        bars = VGroup(*[
            Rectangle(height=2.0, width=0.3, fill_opacity=1, fill_color="#FFFFFF", stroke_width=1, color="#FFFFFF")
            for _ in range(8)
        ]).arrange(RIGHT, buff=0.2)
        
        labels = VGroup(*[
            Text(f"|{bin(i)[2:].zfill(3)}>", font_size=16, color="#FFFFFF")
            for i in range(8)
        ])
        
        for i in range(8):
            labels[i].next_to(bars[i], DOWN, buff=0.2)
            
        graph_group = VGroup(bars, labels)
        # Apply layout refinement from Issue 39: Improve readability of labels
        self.place_in_area(graph_group, "B1", "F6", scale_factor=0.8)
        
        self.lecture[0].set_color(YELLOW)
        self.play(
            Create(bars), 
            Write(labels), 
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Apply a cyan glow (#00FFFF) to all 8 bars simultaneously to represent the state of 'superposition'.
        
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        self.play(
            *[bar.animate.set_fill("#00FFFF").set_stroke("#00FFFF", width=2) for bar in bars],
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Display the text 'Equal Probability' in light grey (#AAAAAA) above the bars as they pulse.
        
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        prob_text = Text("Equal Probability", font_size=24, color="#AAAAAA")
        # Apply layout refinement from Issue 39: Position closer to graph
        self.place_in_area(prob_text, "A2", "A5", scale_factor=0.9)
        
        self.play(FadeIn(prob_text, shift=UP))
        
        # Pulsing effect for bars and text
        for _ in range(2):
            self.play(
                bars.animate.scale(1.1),
                prob_text.animate.scale(1.1),
                rate_func=there_and_back,
                run_time=1.0
            )
        
        self.wait(2)
