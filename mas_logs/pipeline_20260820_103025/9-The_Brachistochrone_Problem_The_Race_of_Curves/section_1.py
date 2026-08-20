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
            "Connect points A and B with a straight line.",
            "A particle slides along a curve under gravity.",
            "Which path reaches point B the fastest?",
            "Straight lines aren't always the fastest path.",
            "Let's discover the Brachistochrone problem."
        ]
        self.setup_layout("Introduction: The Brachistochrone Problem", lecture_lines)
        
        # Assets
        ball = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg")
        incline = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/incline.svg")
        
        dot_a = Dot(color=BLUE)
        dot_b = Dot(color=BLUE)
        self.place_at_grid(dot_a, 'A2', scale_factor=1.0)
        self.place_at_grid(dot_b, 'F5', scale_factor=1.0)
        
        label_a = Text("A", font_size=20).next_to(dot_a, UP)
        label_b = Text("B", font_size=20).next_to(dot_b, DOWN)
        
        line = Line(dot_a.get_center(), dot_b.get_center(), color=GRAY)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(dot_a), FadeIn(dot_b), Write(label_a), Write(label_b))
        self.lecture[0].set_color(YELLOW)
        self.play(Create(line))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        self.place_at_grid(incline, 'C3', scale_factor=0.5)
        self.play(FadeIn(incline))
        
        # Animate ball on incline
        ball_start = incline.get_center() + UP * 0.5
        ball.move_to(ball_start).scale(0.3)
        self.play(FadeIn(ball))
        
        path = ArcBetweenPoints(ball.get_center(), dot_b.get_center(), angle=-TAU/8, color=RED)
        self.play(MoveAlongPath(ball, path), run_time=2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        self.play(Indicate(line))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        self.wait(2)
