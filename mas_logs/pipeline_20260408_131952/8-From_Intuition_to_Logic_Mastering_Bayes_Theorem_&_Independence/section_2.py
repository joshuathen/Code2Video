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
        # Initialize Layout
        title_text = "The Shift: Conditional Probability (Dependency)"
        lines = [
            "Real-world events often provide clues about each other.",
            "Evidence like footprints shrinks our possible sample space.",
            "Conditional probability measures outcomes given specific information."
        ]
        self.setup_layout(title_text, lines)
        
        # Color definitions per instruction
        COLOR_BEAR = "#9B59B6"      # Purple
        COLOR_FOOTPRINT = "#F1C40F" # Yellow
        COLOR_SS = "#ECF0F1"        # Light Grey/White

        # === Animation for Lecture Line 1 ===
        # Matching color: Purple for the first line and the Bear elements
        self.play(self.lecture[0].animate.set_color(COLOR_BEAR))

        # Sample Space Rectangle
        ss_rect = Rectangle(width=5.5, height=5.5, color=COLOR_SS, stroke_width=2)
        self.place_in_area(ss_rect, "A1", "F6")
        
        ss_label = Text("Sample Space", font_size=18, color=COLOR_SS)
        # Fix Issue 31: Better centering for the sample space label
        self.place_in_area(ss_label, 'A2', 'A5')

        # Suspect Bear Circle
        bear_circle = Circle(radius=1.2, color=COLOR_BEAR, fill_opacity=0.3)
        # Fix Issue 32: Scale down circle to accommodate the label without truncation
        self.place_at_grid(bear_circle, "C2", scale_factor=0.8)
        
        bear_label = Text("Suspect: Bear", font_size=20, color=COLOR_BEAR)
        bear_label.next_to(bear_circle, DOWN, buff=0.2)

        self.play(Create(ss_rect), Write(ss_label))
        self.play(Create(bear_circle), Write(bear_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Matching color: Yellow for the second line and the Footprint circle
        self.play(self.lecture[1].animate.set_color(COLOR_FOOTPRINT))

        # Footprint Circle (Overlaps the Bear circle significantly)
        footprint_circle = Circle(radius=1.2, color=COLOR_FOOTPRINT, fill_opacity=0.3)
        # Fix Issue 33: Scale down circle to accommodate the label without truncation
        self.place_at_grid(footprint_circle, "C4", scale_factor=0.8)
        
        footprint_label = Text("Footprint Found", font_size=20, color=COLOR_FOOTPRINT)
        footprint_label.next_to(footprint_circle, DOWN, buff=0.2)

        self.play(Create(footprint_circle), Write(footprint_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Matching color: SS White for the third line
        self.play(self.lecture[2].animate.set_color(COLOR_SS))

        # Visualization of the Shrunken Sample Space
        # Create a blackout mask covering the original sample space except the footprint circle
        mask_base = Rectangle(width=5.5, height=5.5).set_fill(BLACK, opacity=0.8).set_stroke(width=0)
        self.place_in_area(mask_base, "A1", "F6")
        
        # Hole creation using Cutout
        dark_mask = Cutout(
            mask_base,
            footprint_circle.copy().scale(1.02),
            fill_opacity=0.8,
            color=BLACK,
            stroke_width=0
        )
        
        # Fade out background elements and highlight the new sample space
        self.play(
            FadeIn(dark_mask),
            ss_label.animate.set_opacity(0.2),
            bear_label.animate.set_opacity(0.2),
            footprint_circle.animate.set_fill(opacity=0.6).set_stroke(width=4)
        )
        
        self.wait(3)
