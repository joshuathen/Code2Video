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
        self.setup_layout("The Mystery of Rotational Growth", [
            "Exponents usually represent linear growth.",
            "Multiplying by 'i' rotates vectors by ninety degrees.",
            "Repeated rotations create circular motion."
        ])
        
        axes = ComplexPlane().scale(0.7)
        self.place_in_area(axes, 'C3', 'F6', scale_factor=0.6)
        self.add(axes)
        
        dot = Dot(color="#FF5733")
        self.place_at_grid(dot, 'D4', scale_factor=0.5)
        self.add(dot)
        
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        gyroscope = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/gyroscope.svg")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF5733"))
        
        self.place_at_grid(compass, 'B2', scale_factor=0.5)
        self.play(FadeIn(compass))
        
        angle = ValueTracker(0)
        dot.add_updater(lambda d: d.move_to(axes.c2p(np.cos(angle.get_value()), np.sin(angle.get_value()))))
        self.play(angle.animate.set_value(2 * PI), run_time=2)
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#33FF57"))
        
        r = ValueTracker(1)
        dot.add_updater(lambda d: d.move_to(axes.c2p(r.get_value() * np.cos(angle.get_value()), r.get_value() * np.sin(angle.get_value()))))
        self.play(r.animate.set_value(2), run_time=2)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#3357FF"))
        
        self.place_at_grid(gyroscope, 'E5', scale_factor=0.5)
        self.play(FadeIn(gyroscope))
        
        self.play(angle.animate.set_value(4 * PI), r.animate.set_value(0.5), run_time=3)
