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
        lecture_lines = ["The solution is the cycloid.", "A point on a rolling wheel.", "Cycloid beats the straight path.", "Curved path captures kinetic energy.", "Movement accelerates faster initially."]
        self.setup_layout("The Counter-Intuitive Solution: The Cycloid", lecture_lines)
        
        # Assets
        wheel = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wheel.svg")
        ball = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg")
        
        track = Line(LEFT*2.5, RIGHT*2.5, color=WHITE)
        self.place_in_area(track, 'C4', 'E6', scale_factor=0.8)
        self.add(track)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        self.place_at_grid(wheel, 'B3', scale_factor=0.5)
        self.add(wheel)
        self.play(wheel.animate.shift(RIGHT*2), run_time=2)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFD700")
        point = Dot(color="#FFD700")
        point.move_to(wheel.get_center())
        self.add(point)
        # Simplified trace representation
        self.play(Rotate(wheel, PI, about_point=wheel.get_center()), run_time=1)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#7FFF00")
        cycloid = ParametricFunction(lambda t: np.array([t - np.sin(t), -np.cos(t), 0]), t_range=[0, 2*PI], color="#7FFF00")
        self.place_at_grid(cycloid, 'D4', scale_factor=0.4)
        self.play(Create(cycloid))
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFFFF")
        straight_path = Line(LEFT*2, RIGHT*2, color=WHITE)
        self.place_at_grid(straight_path, 'E4', scale_factor=0.4)
        self.play(Create(straight_path))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF4500")
        self.place_at_grid(ball, 'C3', scale_factor=0.3)
        self.play(MoveAlongPath(ball, cycloid), run_time=2)
        self.wait(1)
