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
        # Setup
        title = "The Unstructured Search Problem"
        lines = [
            "Imagine searching an unsorted database of one thousand items.",
            "Classically, we must check every item one by one.",
            "This linear search takes up to one thousand attempts."
        ]
        self.setup_layout(title, lines)
        
        # Colors for lecture steps
        color_step1 = "#87CEEB" # Sky Blue
        color_step2 = "#FFFFE0" # Light Yellow
        color_step3 = "#FFB6C1" # Light Pink

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(color_step1))
        
        # Load database icon [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/database.svg]
        # We handle Issue 20 here.
        db_icon = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/database.svg")
        db_icon.set_color(WHITE)
        self.place_in_area(db_icon, "A1", "F6", scale_factor=1.2)
        
        # Create 1,000 small dark green circles (#556B2F)
        circles = VGroup(*[
            Circle(radius=0.04, color="#556B2F", fill_opacity=1, stroke_width=0.5) 
            for _ in range(1000)
        ])
        circles.arrange_in_grid(rows=25, cols=40, buff=0.06)
        
        # Place the grid in area A1 to F6 with scale factor 0.85 (Fixes Issue 22 and 23)
        self.place_in_area(circles, "A1", "F6", scale_factor=0.85)
        
        self.play(FadeIn(db_icon))
        self.wait(0.5)
        # Database icon expands into grid of circles
        self.play(ReplacementTransform(db_icon, circles), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(color_step2))
        
        # One random circle flashes Gold (#FFD700) to reveal the hidden target
        target_index = 842 
        target_circle = circles[target_index]
        
        self.play(target_circle.animate.set_color("#FFD700").scale(1.5), run_time=0.5)
        self.play(target_circle.animate.scale(1/1.5), run_time=0.5)
        
        # A white highlight box (#FFFFFF) scans the grid circles one-by-one
        scanner = Square(side_length=0.15, color="#FFFFFF", stroke_width=2)
        scanner.move_to(circles[0].get_center())
        
        self.add(scanner)
        
        # Animate scanner for the first few items to show "one by one"
        # Then skip ahead to demonstrate complexity
        scan_indices = list(range(0, 15)) + list(range(15, 80, 5)) 
        for idx in scan_indices:
            self.play(scanner.animate.move_to(circles[idx].get_center()), run_time=0.05)
            
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(color_step3))
        
        # Fast scan towards the end to emphasize "one thousand attempts"
        fast_indices = [200, 400, 600, 800, target_index]
        for idx in fast_indices:
            self.play(scanner.animate.move_to(circles[idx].get_center()), run_time=0.2)
        
        self.play(Indicate(target_circle, color="#FFD700"))
        self.wait(2)
