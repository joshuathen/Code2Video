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
        self.setup_layout("Strategy 1: The Principle of Extremity", [
            "Consider the extreme case first.",
            "Maximize or minimize the objective function.",
            "Boundary analysis collapses complex systems."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg]
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        title_extremity = Text("Principle of Extremity", font_size=36, color="#00FFFF")
        self.place_at_grid(title_extremity, 'A1', scale_factor=1.0) # Fixed Issue 25/40
        
        group1 = VGroup(compass, title_extremity)
        self.play(FadeIn(group1))
        self.lecture[0].set_color("#00FFFF")

        # === Animation for Lecture Line 2 ===
        points = VGroup(*[Dot(color="#FFCC00") for _ in range(20)])
        for p in points:
            p.move_to(self.grid['C3'] + np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0]))
        
        max_point = max(points, key=lambda p: p.get_x())
        max_point.set_color(RED)
        
        anim_group = VGroup(points, max_point)
        self.place_in_area(anim_group, 'C4', 'F6', scale_factor=0.9) # Fixed Issue 23/38
        self.play(Create(points))
        self.play(Indicate(max_point))
        self.lecture[1].set_color("#FFCC00")

        # === Animation for Lecture Line 3 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/scale.svg]
        scale_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scale.svg")
        self.place_at_grid(scale_icon, 'F4', scale_factor=0.5)
        
        self.play(points.animate.set_color("#FFFF00"), FadeIn(scale_icon))
        self.lecture[2].set_color("#FFFFFF")
        
        self.wait(2)
