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
        self.setup_layout("Mapping Moves to the Sierpinski Triangle", [
            "- Legal moves trace a Sierpinski triangle path.",
            "- Each fractal level mirrors the state space.",
            "- The disk path creates a beautiful geometric fractal."
        ])

        # === Animation for Lecture Line 1 ===
        # 1. Fade in the text 'Sierpinski Mapping'
        # Applying requested position/scale fixes (Issues 24, 25, 31)
        mapping_text = Text("Sierpinski Mapping", font_size=24, color=WHITE)
        self.place_at_grid(mapping_text, 'B2', scale_factor=0.6)
        
        # Note: self.title is handled in setup_layout, but requested fixes for line 57 
        # (title anchor/scaling) might refer to mapping_text or similar.
        
        self.play(FadeIn(mapping_text))
        self.play(self.lecture[0].animate.set_color(YELLOW))

        # === Animation for Lecture Line 2 ===
        # 2. Draw the base triangle with Sierpinski pattern
        # Including disk asset [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg]
        disk_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg", color="#00FFFF")

        def get_sierpinski_with_disks(order, size):
            if order == 0:
                return disk_asset.copy().scale_to_fit_width(size)
            
            sub = get_sierpinski_with_disks(order - 1, size / 2)
            t1 = sub.copy()
            t2 = sub.copy()
            t3 = sub.copy()
            
            # Simplified triangle layout
            t1.shift(UP * (size/2) * np.sqrt(3)/2)
            t2.shift(LEFT * (size/2) + DOWN * (size/2) * np.sqrt(3)/2)
            t3.shift(RIGHT * (size/2) + DOWN * (size/2) * np.sqrt(3)/2)
            
            return VGroup(t1, t2, t3)

        sierpinski = get_sierpinski_with_disks(2, 0.8)
        # Applying requested position/scale fixes (Issues 23, 31)
        self.place_in_area(sierpinski, 'B4', 'E6', scale_factor=0.4)
        
        self.play(Create(sierpinski))
        self.play(self.lecture[1].animate.set_color(YELLOW))

        # === Animation for Lecture Line 3 ===
        # 3. Map tower moves onto triangle fractal transitions
        path = VMobject(color=ORANGE, stroke_width=3)
        # Simplified path representing Hanoi move sequence
        points = [sierpinski.get_center() + UP*0.5, sierpinski.get_center() + LEFT*0.5, sierpinski.get_center() + RIGHT*0.5, sierpinski.get_center() + UP*0.5]
        path.set_points_smoothly(points)
        
        self.play(Create(path), run_time=2)
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.wait(2)
