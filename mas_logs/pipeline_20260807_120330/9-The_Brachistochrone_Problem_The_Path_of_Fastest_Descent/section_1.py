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
        # Data from storyboard
        title = "The Racing Paradox"
        lines = [
            "Which path gets a marble from A to B fastest?",
            "A straight line is the shortest distance.",
            "But is it the fastest path?",
            "Let's race marbles on different curves.",
            "Surprisingly, the straight line loses the race."
        ]
        
        self.setup_layout(title, lines)
        
        # === Animation for Lecture Line 1 ===
        # Display Point A and Point B with labels (#FFFFFF).
        # Fix Issue 22: Move Point A to 'B3'
        # Fix Issue 23: Move Point B to 'E6'
        self.lecture[0].set_color(WHITE)
        point_a = Dot(color=WHITE)
        point_b = Dot(color=WHITE)
        self.place_at_grid(point_a, 'B3', scale_factor=1.0)
        self.place_at_grid(point_b, 'E6', scale_factor=1.0)
        
        label_a = MathTex("A", color=WHITE).scale(0.8)
        label_b = MathTex("B", color=WHITE).scale(0.8)
        label_a.next_to(point_a, UP)
        label_b.next_to(point_b, DOWN)
        
        self.play(FadeIn(point_a), FadeIn(point_b), Write(label_a), Write(label_b))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Draw three paths: Straight Line (#FF4444)
        self.lecture[1].set_color("#FF4444")
        
        start_pos = self.grid['B3']
        end_pos = self.grid['E6']
        
        straight_path = Line(start_pos, end_pos, color="#FF4444")
        
        self.play(Create(straight_path))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Deep Curve (#44FF44), and Steepest Curve (#4444FF).
        self.lecture[2].set_color(WHITE) # Neutral/General question
        
        # Deep Curve (Circular/Parabolic approximation)
        deep_path = CubicBezier(
            start_pos,
            start_pos + DOWN * 1.5 + RIGHT * 0.5,
            end_pos + LEFT * 1.5,
            end_pos,
            color="#44FF44"
        )
        
        # Steepest Curve (Cycloid approximation)
        # Brachistochrone starts very vertically
        steep_path = CubicBezier(
            start_pos,
            start_pos + DOWN * 3,
            end_pos + LEFT * 0.5,
            end_pos,
            color="#4444FF"
        )
        
        self.play(Create(deep_path), Create(steep_path))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Start the race: Marbles [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/marble.svg] (#FFFFFF) traverse the paths at different speeds.
        # Fix Issue 19: Use Asset marble.svg
        self.lecture[3].set_color(WHITE)
        
        marble_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/marble.svg"
        
        marble_red = SVGMobject(marble_asset).scale(0.15).set_color(WHITE).move_to(start_pos)
        marble_green = SVGMobject(marble_asset).scale(0.15).set_color(WHITE).move_to(start_pos)
        marble_blue = SVGMobject(marble_asset).scale(0.15).set_color(WHITE).move_to(start_pos)
        
        self.add(marble_red, marble_green, marble_blue)
        
        # We want Blue to win, then Green, then Red.
        # Blue (Steepest) path is the fastest descent.
        self.play(
            MoveAlongPath(marble_blue, steep_path, rate_func=linear, run_time=2.0),
            MoveAlongPath(marble_green, deep_path, rate_func=linear, run_time=2.5),
            MoveAlongPath(marble_red, straight_path, rate_func=linear, run_time=3.5),
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Highlight the Steepest Curve (#4444FF) as it finishes first.
        # Surprisingly, the straight line loses the race.
        self.lecture[4].set_color("#4444FF")
        
        self.play(
            steep_path.animate.set_stroke(width=8),
            marble_blue.animate.scale(1.5),
            Indicate(self.lecture[4])
        )
        self.wait(2)
