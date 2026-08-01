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

class Section1Scene(TeachingScene):
    def construct(self):
        # Initialize title and lecture lines
        title_str = "The Refraction Mystery"
        lines = [
            'Ever noticed light bending as it enters water?',
            'This phenomenon is known as refraction.',
            'Why does light choose this specific path?'
        ]
        self.setup_layout(title_str, lines)
        
        # Colors
        COLOR_AIR = "#FFFFFF"
        COLOR_WATER = "#00BFFF"
        COLOR_RAY = "#FFFF00"
        COLOR_NORMAL = "#A9A9A9"
        
        # Elements
        # Interface line (Horizontal)
        interface = Line(
            start=self.grid['C1'], 
            end=self.grid['C6'], 
            color=WHITE, 
            stroke_width=2
        )
        
        # Labels
        label_air = Text("Air", color=COLOR_AIR, font_size=20)
        self.place_at_grid(label_air, "B5", scale_factor=1.1)
        
        label_water = Text("Water", color=COLOR_WATER, font_size=20)
        self.place_at_grid(label_water, "D5", scale_factor=1.1)
        
        # Normal line
        normal_top = [self.grid['C3'][0], 3.0, 0]
        normal_bottom = [self.grid['C3'][0], -3.0, 0]
        normal_line = DashedLine(
            start=normal_top, 
            end=normal_bottom, 
            color=COLOR_NORMAL
        )
        
        # Points for the ray
        pt_a = self.grid['A2']
        pt_p = self.grid['C3']  # Intersection on interface
        pt_b = self.grid['E4']  # Destination in water (bending towards normal)
        
        ray_air = Arrow(start=pt_a, end=pt_p, color=COLOR_RAY, buff=0, stroke_width=4)
        ray_water = Arrow(start=pt_p, end=pt_b, color=COLOR_RAY, buff=0, stroke_width=4)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Create(interface), FadeIn(label_air), FadeIn(label_water))
        self.play(GrowArrow(ray_air))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(Create(normal_line))
        self.play(GrowArrow(ray_water))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Highlight the ray to draw attention to the path
        self.play(
            ray_air.animate.set_stroke(width=6),
            ray_water.animate.set_stroke(width=6),
            run_time=0.5
        )
        self.play(
            ray_air.animate.set_stroke(width=4),
            ray_water.animate.set_stroke(width=4),
            run_time=0.5
        )
        self.wait(2)
