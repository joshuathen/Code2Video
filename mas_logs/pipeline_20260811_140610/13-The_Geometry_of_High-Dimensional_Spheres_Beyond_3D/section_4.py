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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Application: The Curse of Dimensionality", [
            "This leads to the curse of dimensionality.",
            "In high dimensions, distance metrics lose their meaning.",
            "Data points effectively migrate to the hypersphere's surface."
        ])
        
        # --- Animation for Lecture Line 1 ---
        # Show a high-dimensional scatter plot, color #AABBCC.
        # Use asset /scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg
        sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color="#AABBCC")
        self.place_at_grid(sphere, 'C4', scale_factor=0.7)
        
        dots = VGroup(*[Dot(radius=0.04, color="#AABBCC") for _ in range(50)])
        for dot in dots:
            dot.move_to(sphere.get_center() + np.array([np.random.uniform(-0.8, 0.8), np.random.uniform(-0.8, 0.8), 0]))
        
        self.play(FadeIn(sphere), FadeIn(dots))
        self.play(self.lecture[0].animate.set_color("#AABBCC"))

        # --- Animation for Lecture Line 2 ---
        self.play(self.lecture[1].animate.set_color("#FF0000"))
        # Show distance metrics failing
        distance_label = Text("Distance fails", font_size=20, color="#FF0000")
        self.place_at_grid(distance_label, 'B4', scale_factor=0.8)
        self.play(Write(distance_label))

        # --- Animation for Lecture Line 3 ---
        # Highlight points migrating to surface of sphere, color #EECC33.
        self.play(self.lecture[2].animate.set_color("#EECC33"))
        
        animations = []
        for dot in dots:
            # Move relative to sphere center
            vec = dot.get_center() - sphere.get_center()
            norm = np.linalg.norm(vec)
            if norm > 0:
                new_pos = sphere.get_center() + (vec / norm * 0.8)
                animations.append(dot.animate.set_color("#EECC33").move_to(new_pos))
        self.play(*animations, run_time=2)
        
        label = Text("Points migrate to surface", font_size=20, color="#EECC33")
        self.place_at_grid(label, 'A4', scale_factor=0.9)
        self.play(Write(label))
        self.wait(2)
