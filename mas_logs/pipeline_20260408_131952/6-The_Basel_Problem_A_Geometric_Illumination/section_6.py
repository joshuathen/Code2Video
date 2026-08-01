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

class Section6Scene(TeachingScene):
    def construct(self):
        # Initial Setup
        lines = [
            "A simple sequence of squares reveals a hidden circle.",
            "The Basel problem shows the deep unity of math.",
            "One plus one-fourth plus one-ninth equals pi-squared over six."
        ]
        self.setup_layout("Summary and Elegance", lines)

        # Pre-create Mobjects
        # Note: Using colors aligned with lecture segments to meet constraints
        basel_lhs = Text("1 + 1/4 + 1/9 + ...", color=BLUE_C, font_size=42)
        basel_rhs = Text("= π²/6", color=YELLOW_C, font_size=42)
        
        # Issue 42: Repositioning formula in Row B to avoid occlusion/overlap
        self.place_in_area(basel_lhs, 'B2', 'B3')
        self.place_in_area(basel_rhs, 'B4', 'B5')

        # Issue 29: Integrate Pixel the Robot Asset
        # Issue 49: Using provided asset path
        pixel = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/robot.svg")
        pixel.set_color("#00FF00")
        
        # Issue 44: Robot at grid F6 to avoid halo circle overlap
        self.place_at_grid(pixel, 'F6', scale_factor=0.8)

        # Issue 43: Halo moved to D2-F5 area to avoid formula overlap
        halo = Circle(radius=1.0, color="#FFFFFF", stroke_width=3)
        self.place_in_area(halo, 'D2', 'F5')

        # === Animation for Lecture Line 1 ===
        # Matching color for Line 1 and the lhs of the formula
        self.lecture[0].set_color(BLUE_C)
        self.play(Write(basel_lhs), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Matching color for Line 2 and the robot
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00FF00") 
        self.play(
            FadeIn(pixel, shift=UP),
            run_time=1.2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Matching color for Line 3 and the result (rhs)
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW_C)
        
        # Show final equation and the halo of light (#FFFFFF per description)
        self.play(
            Write(basel_rhs),
            Create(halo),
            run_time=2
        )

        # Visual Emphasis (Final glow and emphasis)
        self.play(
            halo.animate.set_stroke(width=10, opacity=0.9),
            pixel.animate.scale(1.2),
            rate_func=there_and_back,
            run_time=2
        )

        self.wait(3)
