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
            "What is the fastest path between two points?",
            "A straight line is shortest, but is it fastest?",
            "Gravity accelerates objects differently on various curves.",
            "Let's compare a straight line and a steep curve.",
            "Surprisingly, the longer, steeper path wins the race."
        ]
        self.setup_layout("The Counter-Intuitive Race", lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        point_a = Dot(self.grid["A2"], color="#FFFFFF")
        label_a = Text("A", font_size=20, color="#FFFFFF").next_to(point_a, UP, buff=0.1)
        point_b = Dot(self.grid["E5"], color="#FFFFFF")
        label_b = Text("B", font_size=20, color="#FFFFFF").next_to(point_b, DOWN, buff=0.1)
        
        self.play(Create(point_a), Create(label_a), Create(point_b), Create(label_b))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00FF00")
        
        straight_track = Line(self.grid["A2"], self.grid["E5"], color="#00FF00")
        label_straight = Text("Shortest", font_size=16, color="#00FF00").next_to(straight_track, UR, buff=-0.5)
        
        self.play(Create(straight_track), FadeIn(label_straight))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#0000FF")
        
        # Circular Track
        circular_track = ArcBetweenPoints(self.grid["A2"], self.grid["E5"], radius=4, color="#0000FF")
        
        # Steep Track (Cycloid-like shape using CubicBezier)
        # We want it to go down fast then move towards B
        start = self.grid["A2"]
        end = self.grid["E5"]
        # Control points to make it steep
        c1 = start + DOWN * 4
        c2 = end + LEFT * 2
        steep_track = CubicBezier(start, c1, c2, end, color="#FF0000")
        
        self.play(Create(circular_track), Create(steep_track))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Three Spheres
        sphere_straight = Dot(self.grid["A2"], color="#FFFFFF", radius=0.12)
        sphere_circular = Dot(self.grid["A2"], color="#FFFFFF", radius=0.12)
        sphere_steep = Dot(self.grid["A2"], color="#FFFFFF", radius=0.12)
        
        self.add(sphere_straight, sphere_circular, sphere_steep)
        
        # Race using ValueTracker for relative timing
        race_time = ValueTracker(0)
        
        # Steep: 1.5s, Straight: 2.2s, Circular: 2.8s
        sphere_steep.add_updater(lambda m: m.move_to(steep_track.point_from_proportion(min(race_time.get_value() / 1.5, 1))))
        sphere_straight.add_updater(lambda m: m.move_to(straight_track.point_from_proportion(min(race_time.get_value() / 2.2, 1))))
        sphere_circular.add_updater(lambda m: m.move_to(circular_track.point_from_proportion(min(race_time.get_value() / 2.8, 1))))
        
        self.play(race_time.animate.set_value(2.8), run_time=3, rate_func=linear)
        
        sphere_steep.clear_updaters()
        sphere_straight.clear_updaters()
        sphere_circular.clear_updaters()
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FF0000")
        
        # Highlight winner
        winner_text = Text("Fastest Path", font_size=28, color="#FF0000")
        # ISSUE 23 FIX: Move to D6 and scale 0.7
        self.place_at_grid(winner_text, "D6", scale_factor=0.7)
        
        highlight_rect = SurroundingRectangle(winner_text, color=YELLOW, buff=0.2)
        
        self.play(Indicate(sphere_steep, color="#FF0000", scale_factor=2.0))
        self.play(Flash(sphere_steep, color="#FF0000", line_length=0.3))
        self.play(Write(winner_text), Create(highlight_rect))
        self.wait(2)
