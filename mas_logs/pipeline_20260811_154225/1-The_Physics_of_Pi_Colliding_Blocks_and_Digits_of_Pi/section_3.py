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
        self.setup_layout("Mapping Collisions to Circular Arcs", [
            "Large mass ratios constrain reflection paths.",
            "Velocity vectors trace chords on a circle.",
            "More mass means smoother circular arcs."
        ])
        
        # Elements as per storyboard and asset requirements
        circle = Circle(radius=1.5, color=WHITE)
        arc = Arc(start_angle=PI/4, angle=PI/2, radius=1.5, color="#1E90FF", stroke_width=6)
        chord = Line(arc.get_start(), arc.get_end(), color="#FFD700")
        
        # SVG Assets
        # Note: None.svg placeholder used as per instructions
        icon1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        icon2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")

        # Grouping for constraints (Applying B004: cols 4-6)
        # Applying B020: Text labels (not applicable here, but objects are managed)
        
        # === Animation for Lecture Line 1 ===
        # Fix: center-heavy composition and grid alignment (Issues 27, 41)
        self.place_in_area(circle, 'C4', 'E6', scale_factor=0.6)
        self.place_at_grid(icon1, 'B4', scale_factor=0.5)
        self.play(Create(circle), FadeIn(icon1), self.lecture[0].animate.set_color("#FFFFFF"))
        
        # === Animation for Lecture Line 2 ===
        # Fix: Label crowding and grid usage (Issues 28, 42)
        self.place_at_grid(arc, 'C5', scale_factor=0.5)
        self.place_at_grid(chord, 'D5', scale_factor=0.5)
        self.play(Create(arc), Create(chord), self.lecture[1].animate.set_color("#1E90FF"))
        
        # === Animation for Lecture Line 3 ===
        smooth_arc = Arc(start_angle=PI/4, angle=PI/2, radius=1.5, color="#00CED1", stroke_width=8)
        self.place_at_grid(smooth_arc, 'E5', scale_factor=0.5)
        self.place_at_grid(icon2, 'F5', scale_factor=0.5)
        self.play(Transform(arc, smooth_arc), FadeIn(icon2), self.lecture[2].animate.set_color("#00CED1"))
        self.wait(1)
