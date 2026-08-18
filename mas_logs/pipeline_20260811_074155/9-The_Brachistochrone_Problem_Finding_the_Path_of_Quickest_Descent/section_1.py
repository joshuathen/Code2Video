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
        title = "The Challenge: The Intuition Trap"
        lines = [
            "What curve gives the fastest descent between two points?",
            "Intuition suggests a straight line is the quickest path.",
            "However, the straight line is only the shortest distance.",
            "A steeper initial drop allows for much higher speeds.",
            "Watch as the \"mystery curve\" beats the straight line."
        ]
        self.setup_layout(title, lines)

        # Positions
        pos_a = self.grid["A2"]
        pos_b = self.grid["E5"]

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        point_a = Dot(pos_a, color=WHITE)
        label_a = Text("A", font_size=20, color=WHITE).next_to(point_a, UP, buff=0.1)
        point_b = Dot(pos_b, color=WHITE)
        label_b = Text("B", font_size=20, color=WHITE).next_to(point_b, DOWN, buff=0.1)
        
        self.play(FadeIn(point_a, label_a, point_b, label_b))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GREEN)
        straight_path = Line(pos_a, pos_b, color=GREEN)
        self.play(Create(straight_path))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        dist_label = Text("Shortest Distance", font_size=18, color=GREEN)
        # Resolved Issue 25: Moved dist_label to D6 to avoid path overlap
        self.place_at_grid(dist_label, "D6", scale_factor=0.8) 
        self.play(Write(dist_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(YELLOW)
        # Circular path (approximation)
        circular_path = ArcBetweenPoints(pos_a, pos_b, radius=4, color=YELLOW)
        
        # Mystery Curve (Cycloid approximation using a Bezier)
        # Brachistochrone drops faster initially
        mid_point = pos_a + (pos_b - pos_a) * 0.4 + DOWN * 1.5
        mystery_path = CubicBezier(pos_a, pos_a + DOWN * 2, mid_point, pos_b, color=PURPLE) 
        
        self.play(Create(circular_path), Create(mystery_path))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(PURPLE)
        
        # Penguins as Dots
        penguin_straight = Dot(color=GREEN).scale(1.2)
        penguin_circular = Dot(color=YELLOW).scale(1.2)
        penguin_mystery = Dot(color=PURPLE).scale(1.2)

        # Race trackers
        self.add(penguin_straight, penguin_circular, penguin_mystery)
        
        self.play(
            MoveAlongPath(penguin_mystery, mystery_path, rate_func=linear, run_time=1.5),
            MoveAlongPath(penguin_circular, circular_path, rate_func=linear, run_time=2.2),
            MoveAlongPath(penguin_straight, straight_path, rate_func=linear, run_time=2.8),
        )
        
        time_label = Text("Shortest Time?", font_size=20, color=PURPLE)
        # Resolved Issue 26: Moved time_label to B6 to avoid overlap with initial drop
        self.place_at_grid(time_label, "B6", scale_factor=0.8)
        
        self.play(
            Flash(mystery_path, color=PURPLE, line_length=0.3, num_lines=12),
            Write(time_label)
        )
        
        self.wait(2)
