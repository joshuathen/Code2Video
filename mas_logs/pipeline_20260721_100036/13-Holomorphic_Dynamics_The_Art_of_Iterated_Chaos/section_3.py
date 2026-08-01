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
        title_text = "Stability vs. Chaos: Fatou and Julia Sets"
        lecture_lines = [
            "Fatou sets contain points with stable, predictable paths.",
            "Nearby points in these regions stay together.",
            "Julia sets form the chaotic boundaries between them.",
            "On the boundary, tiny changes lead to chaos.",
            "This edge of chaos creates intricate beauty."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Assets/Colors
        color_fatou = "#ADD8E6"
        color_julia = "#FFD700"
        color_dot = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Fatou sets contain points with stable, predictable paths.
        self.lecture[0].set_color(color_fatou)
        
        # Create a stylized "Fatou region" (smooth circle-ish area)
        fatou_region = Circle(radius=1.5, color=color_fatou, fill_opacity=0.1)
        self.place_in_area(fatou_region, "B2", "E5")
        
        fatou_label = Text("Fatou Set", font_size=20, color=color_fatou)
        # Fix Issue 24: Move to A3 and scale down to avoid overlap
        self.place_at_grid(fatou_label, "A3", scale_factor=0.8)
        
        self.play(Create(fatou_region), Write(fatou_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Nearby points in these regions stay together.
        self.lecture[1].set_color(color_fatou)
        
        dot1 = Dot(color=color_dot, radius=0.08)
        dot2 = Dot(color=color_dot, radius=0.08)
        
        # Positions inside Fatou region
        start_pos = self.grid["D3"]
        dot1.move_to(start_pos + LEFT*0.1)
        dot2.move_to(start_pos + RIGHT*0.1)
        
        # Path for Fatou (stable)
        path1 = Line(dot1.get_center(), self.grid["E3"] + LEFT*0.1, color=color_fatou, stroke_width=2)
        path2 = Line(dot2.get_center(), self.grid["E3"] + RIGHT*0.1, color=color_fatou, stroke_width=2)
        
        self.play(FadeIn(dot1, dot2))
        self.play(
            MoveAlongPath(dot1, path1),
            MoveAlongPath(dot2, path2),
            Create(path1),
            Create(path2),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Julia sets form the chaotic boundaries between them.
        self.lecture[2].set_color(color_julia)
        
        # Represent Julia Set as a jagged boundary
        julia_boundary = RegularPolygram(num_vertices=100, radius=1.6, color=color_julia)
        # Add some "fractal" noise to the boundary
        points = julia_boundary.get_points()
        noise = np.random.uniform(-0.1, 0.1, size=points.shape)
        julia_boundary.set_points(points + noise)
        self.place_in_area(julia_boundary, "B2", "E5")
        
        julia_label = Text("Julia Set", font_size=20, color=color_julia)
        # Fix Issue 25: Move to C6 and scale down to avoid clutter
        self.place_at_grid(julia_label, "C6", scale_factor=0.8)
        
        self.play(Create(julia_boundary), Write(julia_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # On the boundary, tiny changes lead to chaos.
        self.lecture[3].set_color(color_julia)
        
        # Move dots to the boundary
        boundary_point = julia_boundary.get_start()
        dot_chaos1 = Dot(color=color_dot, radius=0.08).move_to(boundary_point)
        dot_chaos2 = Dot(color=color_dot, radius=0.08).move_to(boundary_point + UP*0.001) # Extremely close
        
        self.play(FadeOut(dot1, dot2, path1, path2))
        self.play(FadeIn(dot_chaos1, dot_chaos2))
        
        # Diverging chaotic paths
        # Path 1: Spirals inward
        # Path 2: Wildly exits
        chaos_path1 = ArcBetweenPoints(dot_chaos1.get_center(), self.grid["D4"], angle=PI/2, color=color_julia)
        chaos_path2 = CubicBezier(dot_chaos2.get_center(), self.grid["B5"], self.grid["A1"], self.grid["F6"], color=color_julia)
        
        self.play(
            MoveAlongPath(dot_chaos1, chaos_path1),
            MoveAlongPath(dot_chaos2, chaos_path2),
            Create(chaos_path1),
            Create(chaos_path2),
            run_time=3
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This edge of chaos creates intricate beauty.
        self.lecture[4].set_color(color_julia)
        
        # Flash the boundary to emphasize "beauty"
        self.play(
            julia_boundary.animate.set_stroke(width=6),
            Flash(julia_label, color=color_julia),
            run_time=1
        )
        self.play(julia_boundary.animate.set_stroke(width=2))
        self.wait(2)
