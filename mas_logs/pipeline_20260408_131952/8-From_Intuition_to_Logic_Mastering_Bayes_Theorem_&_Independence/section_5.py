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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup layout with specific title and lecture lines
        self.setup_layout(
            "Summary: The Bayesian Mindset", 
            [
                "Bayes' Theorem is a tool for updating our worldview.", 
                "Turn initial hypotheses into evidence-based conclusions.", 
                "Master the logic of uncertainty for better data science."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.lecture[0].set_color(YELLOW)
        
        # Draw a horizontal sliding scale with 'Initial Guess' on the left and 'Truth' on the right (#ECF0F1)
        scale_color = "#ECF0F1"
        scale_line = Line(
            start=self.grid['D1'], 
            end=self.grid['D6'], 
            color=scale_color, 
            stroke_width=4
        )
        
        # Add decorative ticks to the scale
        ticks = VGroup(*[
            Line(UP*0.1, DOWN*0.1, color=scale_color).move_to(self.grid[f'D{i}']) 
            for i in range(1, 7)
        ])
        
        # Labels for the scale - Fixed positioning to avoid edge bleed/clipping
        label_left = Text("Initial Guess", font_size=18, color=scale_color)
        self.place_in_area(label_left, 'E1', 'E2', scale_factor=0.8)
        
        label_right = Text("Updated Truth", font_size=18, color=scale_color)
        self.place_in_area(label_right, 'E5', 'E6', scale_factor=0.8)
        
        # Pointer (#E67E22) starting at the left
        pointer_color = "#E67E22"
        pointer = Triangle(color=pointer_color, fill_opacity=1).scale(0.15).rotate(PI)
        pointer.move_to(self.grid['D1'] + UP * 0.4)
        
        self.play(
            Create(scale_line),
            Create(ticks),
            Write(label_left),
            Write(label_right),
            FadeIn(pointer)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Update highlighting
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Animate a yellow 'Evidence' icon (#F1C40F) falling onto the scale
        evidence_color = "#F1C40F"
        evidence_circle = Circle(radius=0.25, color=evidence_color, fill_opacity=1)
        evidence_label = Text("E", font_size=16, color=BLACK, weight=BOLD)
        evidence_icon = VGroup(evidence_circle, evidence_label)
        
        # Initial position of evidence (falling from top) - Fixed height
        self.place_at_grid(evidence_icon, 'B3', scale_factor=0.8)
        
        # Evidence falls and pointer moves towards right
        self.play(
            evidence_icon.animate.move_to(self.grid['D3'] + UP * 0.4),
            run_time=1.5,
            rate_func=linear
        )
        
        # Pointer moves to reflect the updated evidence
        self.play(
            pointer.animate.move_to(self.grid['D4'] + UP * 0.4),
            FadeOut(evidence_icon, shift=DOWN),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Update highlighting
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Move the scale's pointer (#E67E22) further towards the right based on accumulation of data
        self.play(
            pointer.animate.move_to(self.grid['D6'] + UP * 0.4),
            run_time=2
        )
        
        # Final emphasis glow on the pointer
        self.play(pointer.animate.scale(1.2).set_color(WHITE), run_time=0.5)
        self.play(pointer.animate.scale(1/1.2).set_color(pointer_color), run_time=0.5)
        
        self.wait(2)
