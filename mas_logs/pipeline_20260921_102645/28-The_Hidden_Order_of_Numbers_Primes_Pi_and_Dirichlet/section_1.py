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
        lecture_lines = [
            "Prime numbers act as mathematical atoms.",
            "They appear chaotic at first glance.",
            "Hidden structures emerge upon deeper inspection.",
            "We visualize them in a number orchard.",
            "Pattern-seeking reveals their underlying order."
        ]
        self.setup_layout("Introduction: The Chaos of Primes", lecture_lines)
        
        # Assets
        orchard_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/orchard.svg"
        
        # Points setup (30 icons)
        point_icons = VGroup(*[SVGMobject(orchard_asset, color=WHITE) for _ in range(30)])
        
        # Initial chaotic positioning in the designated area
        prime_cloud = VGroup()
        for i in range(30):
            p = point_icons[i]
            p.move_to(np.array([np.random.uniform(2, 6), np.random.uniform(-2, 2), 0]))
            prime_cloud.add(p)
        
        # Primes (first 10 primes/indices)
        prime_indices = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        
        # Fix: Constrain prime_cloud as per review
        self.place_in_area(prime_cloud, 'A2', 'F5', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(prime_cloud))
        self.play(self.lecture[0].animate.set_color("#FF4500"))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF4500"))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        for i in prime_indices:
            self.play(point_icons[i].animate.set_color("#FF4500"), run_time=0.1)
            
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFFF00"))
        # Grid arrangement
        animations = []
        for i, pos in enumerate(self.grid.values()):
            if i < 30:
                animations.append(point_icons[i].animate.move_to(pos))
        self.play(*animations)
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#32CD32"))
        
        trace_path = VGroup()
        for i in range(len(prime_indices)-1):
            line = Line(point_icons[prime_indices[i]].get_center(), point_icons[prime_indices[i+1]].get_center(), color="#32CD32", stroke_width=2)
            trace_path.add(line)
        
        # Fix: Apply trace_path constraint
        self.place_in_area(trace_path, 'C2', 'F5', scale_factor=0.5)
        
        self.play(Create(trace_path))
        self.wait(2)
