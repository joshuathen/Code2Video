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
        self.setup_layout("The Brachistochrone Problem", [
            "What is the fastest path between two points?",
            "A straight line is not always the quickest route.",
            "Gravity accelerates objects along this downhill journey."
        ])

        # Assets
        particle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/particle.svg", color=WHITE)
        sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color=WHITE)
        
        point_1 = sphere.copy()
        point_2 = sphere.copy()
        self.place_at_grid(point_1, 'B4', scale_factor=0.7)
        self.place_at_grid(point_2, 'E5', scale_factor=0.7)
        
        straight_path = Line(point_1.get_center(), point_2.get_center(), color=WHITE)
        
        # Brachistochrone curve
        brachistochrone_curve = ParametricFunction(
            lambda t: np.array([
                2.5 * (t - np.sin(t)) - 2.5,
                -2.5 * (1 - np.cos(t)) + 1.5,
                0
            ]),
            t_range=[0, PI],
            color=WHITE
        )
        self.place_in_area(brachistochrone_curve, 'A4', 'F6', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(straight_path), FadeIn(point_1), FadeIn(point_2))
        self.lecture[0].set_color("#00CED1")
        self.play(point_1.animate.set_color("#00CED1"), point_2.animate.set_color("#00CED1"))

        # === Animation for Lecture Line 2 ===
        self.play(ReplacementTransform(straight_path, brachistochrone_curve))
        self.lecture[1].set_color("#FFD700")
        self.play(brachistochrone_curve.animate.set_color("#FFD700"))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF4500")
        self.play(FadeIn(particle.move_to(point_1.get_center())))
        self.play(MoveAlongPath(particle, brachistochrone_curve), run_time=2)
        self.wait(1)
