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
        self.setup_layout("The Brachistochrone Problem: The Race", [
            "What path minimizes a bead's descent time?", 
            "Compare a straight line against an arc.", 
            "Observe the race between three different paths."
        ])
        
        # Define Paths
        start = np.array([-1, 1.5, 0])
        end = np.array([1.5, -1, 0])
        
        # Straight line
        line = Line(start, end, color=BLUE)
        
        # Arc (Approximation)
        arc = ArcBetweenPoints(start, end, angle=-PI/3, color=GREEN)
        
        # Brachistochrone (Cycloid approximation)
        curve = ParametricFunction(
            lambda t: np.array([
                1.5 * (t - np.sin(t)),
                -1.5 * (1 - np.cos(t)),
                0
            ]), t_range=[0, 1.5], color=RED
        )
        # Scale/Translate curve to match endpoints roughly
        curve.scale(0.8).move_to(np.array([0.25, 0.25, 0]))

        # Group them
        group = VGroup(line, arc, curve)
        
        # Apply positioning per fix recommendation (B002)
        self.place_in_area(group, 'B4', 'E6', scale_factor=0.6)
        
        # Add Beads
        dot1 = Dot(color=BLUE).move_to(line.get_start())
        dot2 = Dot(color=GREEN).move_to(arc.get_start())
        dot3 = Dot(color=RED).move_to(curve.get_start())
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(Create(line), Create(arc), Create(curve))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.add(dot1, dot2, dot3)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(RED))
        self.play(
            MoveAlongPath(dot1, line, run_time=2, rate_func=linear),
            MoveAlongPath(dot2, arc, run_time=2, rate_func=linear),
            MoveAlongPath(dot3, curve, run_time=1.5, rate_func=linear)
        )
        self.wait(1)
