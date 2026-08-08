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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Conclusion: The Tautochrone Connection", [
            "Tautochrone: same time from any start.",
            "Nature favors efficient energy transfer.",
            "The cycloid is nature's fastest path."
        ])

        # Cycloid param: x = a(theta - sin(theta)), y = -a(1 - cos(theta))
        a = 0.8
        def cycloid_path(t):
            return np.array([a * (t - np.sin(t)), -a * (1 - np.cos(t)), 0])

        curve = ParametricFunction(cycloid_path, t_range=[0, 2 * np.pi], color=BLUE)
        self.place_in_area(curve, 'C3', 'F6', scale_factor=0.8)
        self.add(curve)

        # SVG asset for balls: /scratch/pawsey1357/jthen/Code2Video/assets/icon/balls.svg
        beads = VGroup(*[SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/balls.svg", color=YELLOW) for _ in range(3)])
        
        start_ts = [np.pi/4, np.pi/2, 3*np.pi/4]
        for i, bead in enumerate(beads):
            bead.scale(0.3)
            bead.move_to(curve.point_from_proportion(start_ts[i] / (2*np.pi)))

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(beads))
        
        # Animate beads sliding down simultaneously
        end_point = curve.point_from_proportion(1.0)
        self.play(*[bead.animate.move_to(end_point) for bead in beads], run_time=2)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(RED))
        
        tautochrone_label = Text("Tautochrone: Same Time", font_size=24, color=YELLOW)
        self.place_at_grid(tautochrone_label, 'B2', scale_factor=0.7)
        self.play(FadeIn(tautochrone_label))
        self.wait(2)
