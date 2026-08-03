from manim import *
import numpy as np
import os

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

class Section7Scene(TeachingScene):
    def construct(self):
        # Title and Lecture Lines from Storyboard
        title_text = "Conclusion: The Universal Language"
        lecture_lines = [
            "- Linear block collisions hide a secret circular nature.",
            "- Conservation laws naturally link physics to the constant Pi.",
            "- Mathematics provides a universal bridge across different worlds."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Colors for matching elements
        COLOR_1 = "#FFFF00"  # Yellow
        COLOR_2 = "#58ACFA"  # Light Blue
        COLOR_3 = "#82FA58"  # Light Green

        # === Animation for Lecture Line 1 ===
        # "Linear block collisions hide a secret circular nature."
        self.play(self.lecture[0].animate.set_color(COLOR_1), run_time=0.5)
        
        # Asset: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg]
        asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg"
        
        # Robust loading of the block asset with fallback to Squares
        blocks_loaded = False
        if os.path.exists(asset_path):
            try:
                blocks = SVGMobject(asset_path)
                if blocks.get_num_points() > 0:
                    blocks.set_height(0.7)
                    blocks_loaded = True
            except:
                blocks_loaded = False
        
        if not blocks_loaded:
            blocks = VGroup(
                Square(side_length=0.3, color=WHITE, fill_opacity=0.5),
                Square(side_length=0.5, color=WHITE, fill_opacity=0.5)
            ).arrange(RIGHT, buff=0.1)
        
        # Issue 36: Position blocks at B4 and scale to 1.2
        self.place_at_grid(blocks, "B4", scale_factor=1.2)
        
        # The hidden geometric circle
        secret_circle = Circle(radius=1.5, color=COLOR_1, stroke_width=4)
        # Issue 38: Position circle in B3-E6 area and scale to 0.9
        self.place_in_area(secret_circle, "B3", "E6", scale_factor=0.9)
        
        # Initial animation: Fade in blocks and draw the circle
        self.play(
            FadeIn(blocks),
            Create(secret_circle),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Conservation laws naturally link physics to the constant Pi."
        # Reset color of previous line and highlight current line
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_2),
            run_time=0.5
        )
        
        # Trace digits of Pi around the circle
        pi_str = "3.14159"
        pi_digits = VGroup(*[Text(d, font_size=24, color=COLOR_2) for d in pi_str])
        
        # Position digits around the perimeter of the circle
        center = secret_circle.get_center()
        radius = secret_circle.width / 2 + 0.2
        for i, digit in enumerate(pi_digits):
            angle = i * (TAU / len(pi_digits))
            pos = center + radius * np.array([np.cos(angle), np.sin(angle), 0])
            digit.move_to(pos)
            
        # Animation: pi digits appearing
        self.play(FadeIn(pi_digits, shift=UP), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Mathematics provides a universal bridge across different worlds."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_3),
            run_time=0.5
        )
        
        # Display summary text in the center
        concl_msg = Text("Physics is Geometry", font_size=30, color=COLOR_3)
        # Issue 37: Position message in C4-D5 area and scale to 0.7
        self.place_in_area(concl_msg, "C4", "D5", scale_factor=0.7)
        
        # Box for emphasis
        box = SurroundingRectangle(concl_msg, color=COLOR_3, buff=0.2)
        
        self.play(
            Write(concl_msg),
            Create(box),
            run_time=2
        )
        self.wait(3)
        
        # Final state: reset lecture line color
        self.play(self.lecture[2].animate.set_color(WHITE), run_time=0.5)
        self.wait(2)
