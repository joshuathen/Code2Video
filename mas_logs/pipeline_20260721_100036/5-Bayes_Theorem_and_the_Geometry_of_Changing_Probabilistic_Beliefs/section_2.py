from manim import *

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
        # Section title and lecture lines from shared state
        title = "Prerequisite: The Unit Square as the Universal Set"
        lecture_lines = [
            "We visualize all possibilities inside a unit square.",
            "The square's total area represents a probability of one.",
            "We split it vertically to show the Phoenix's rarity."
        ]
        self.setup_layout(title, lecture_lines)

        # Color constants
        COLOR_H = "#0000FF"      # Blue for the H strip
        COLOR_NOT_H = "#AAAAAA"  # Gray for not H
        COLOR_SQUARE = "#FFFFFF" # White for the square outline
        COLOR_ACTIVE = "#FFFF00" # Yellow to highlight active lines/concepts

        # === Animation for Lecture Line 1 ===
        # "We visualize all possibilities inside a unit square."
        self.play(self.lecture[0].animate.set_color(COLOR_SQUARE))
        
        # Create a square that fits the B2-E5 area (3x3 grid units)
        square_outline = Rectangle(width=3, height=3, color=COLOR_SQUARE, stroke_width=2)
        self.place_in_area(square_outline, "B2", "E5")
        
        self.play(Create(square_outline))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "The square's total area represents a probability of one."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_ACTIVE)
        )
        
        # Add a label indicating the total area is 1.0
        label_total_area = MathTex("Area = 1.0", font_size=32, color=WHITE)
        # FIXED: Issue 24 - Positioned in area A3-A4 to avoid overlap and fit length
        self.place_in_area(label_total_area, "A3", "A4", scale_factor=0.8)
        
        self.play(Write(label_total_area))
        self.play(square_outline.animate.set_stroke(width=4)) # Briefly emphasize the boundary
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "We split it vertically to show the Phoenix's rarity."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_H)
        )
        
        # Strip width for 'H' (representing a thin sliver)
        # Visual width 0.1 out of 3.0 total width
        h_width = 0.1
        not_h_width = 3.0 - h_width
        
        h_rect = Rectangle(
            width=h_width, 
            height=3, 
            fill_color=COLOR_H, 
            fill_opacity=0.8, 
            stroke_width=1, 
            color=COLOR_H
        )
        not_h_rect = Rectangle(
            width=not_h_width, 
            height=3, 
            fill_color=COLOR_NOT_H, 
            fill_opacity=0.4, 
            stroke_width=1, 
            color=COLOR_NOT_H
        )
        
        # Group them to maintain the 3x3 footprint and align them
        split_square = VGroup(h_rect, not_h_rect).arrange(RIGHT, buff=0)
        self.place_in_area(split_square, "B2", "E5")
        
        # Labels for the regions
        # FIXED: Issue 20 - Integrated SVG asset for the 'H' label
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/pho.svg
        pho_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pho.svg").scale(0.3).set_color(COLOR_H)
        label_h_text = MathTex("H", color=COLOR_H, font_size=36)
        label_h = VGroup(label_h_text, pho_icon).arrange(DOWN, buff=0.1)
        
        # H is tiny, so label it at the top of col 2 (A2)
        self.place_at_grid(label_h, "A2", scale_factor=0.8)
        
        # Not H is large, label it in the center of the remaining area (approx D4)
        label_not_h = MathTex("\\neg H", color=COLOR_NOT_H, font_size=40)
        # FIXED: Issue 25 - Scale factor consistent with label_h logic
        self.place_at_grid(label_not_h, "D4", scale_factor=1.0)
        
        # Transition: Remove total area label, Fade in the regions
        self.play(FadeOut(label_total_area))
        self.play(
            FadeIn(h_rect),
            FadeIn(not_h_rect),
            square_outline.animate.set_stroke(opacity=0.3)
        )
        self.play(Write(label_h), Write(label_not_h))
        self.wait(3)
        
        # Cleanup colors for lecture lines
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
