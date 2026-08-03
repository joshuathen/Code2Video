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
        title = "The Paradox of the Straight Line"
        lecture_lines = [
            "A straight line is the shortest distance.",
            "However, speed is low at the beginning.",
            "A steep curve builds speed much faster.",
            "Higher speed covers more distance in less time.",
            "Speed beats distance in this race."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors
        straight_color = "#FFFFFF"
        curve_color = "#00BFFF"
        bicycle_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/bicycle.svg"

        # Points
        start_point_pos = self.grid["B1"]
        end_point_pos = self.grid["D5"]

        start_dot = Dot(start_point_pos, color=WHITE)
        end_dot = Dot(end_point_pos, color=WHITE)
        start_label = Text("Start", font_size=18).next_to(start_dot, UP, buff=0.1)
        end_label = Text("End", font_size=18).next_to(end_dot, RIGHT, buff=0.1)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        straight_line = Line(start_point_pos, end_point_pos, color=straight_color)
        
        # Load bicycle assets for each path
        bike_straight = SVGMobject(bicycle_path).scale(0.15).move_to(start_point_pos).set_color(straight_color)
        bike_curve = SVGMobject(bicycle_path).scale(0.15).move_to(start_point_pos).set_color(curve_color)

        self.play(FadeIn(start_dot, end_dot, start_label, end_label))
        self.play(Create(straight_line))
        self.play(FadeIn(bike_straight))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Velocity vector on straight line (Low speed)
        direction = (end_point_pos - start_point_pos) / np.linalg.norm(end_point_pos - start_point_pos)
        v_low_pos = start_point_pos + direction * 0.5
        v_low_arrow = Arrow(v_low_pos, v_low_pos + direction * 0.4, buff=0, color=straight_color)
        v_low_label = Text("Slow", font_size=16, color=straight_color).next_to(v_low_arrow, UP, buff=0.1)
        
        self.play(Create(v_low_arrow), Write(v_low_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Define a steep curve
        steep_curve = CubicBezier(
            start_point_pos, 
            self.grid["D1"], 
            self.grid["D3"], 
            end_point_pos, 
            color=curve_color
        )
        
        # Velocity vector on curve (High speed)
        v_high_pos = steep_curve.point_from_proportion(0.15)
        # Numerical approximation of tangent for arrow direction
        dt = 0.01
        p1 = steep_curve.point_from_proportion(0.15)
        p2 = steep_curve.point_from_proportion(0.15 + dt)
        direction_high = (p2 - p1) / np.linalg.norm(p2 - p1)
        
        v_high_arrow = Arrow(v_high_pos, v_high_pos + direction_high * 1.2, buff=0, color=curve_color)
        
        # Create 'Fast' label and position it according to Critic fix (Issue 34)
        v_high_label = Text("Fast", font_size=16, color=curve_color)
        self.place_at_grid(v_high_label, 'E1', scale_factor=0.7)
        
        self.play(Create(steep_curve), FadeIn(bike_curve))
        self.play(Create(v_high_arrow), Write(v_high_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Move bicycles along their paths
        # Curve is faster initially (linear rate for simplicity in comparison to delayed straight)
        # Straight line starts slow and accelerates (t**2)
        self.play(
            MoveAlongPath(bike_curve, steep_curve, rate_func=linear, run_time=1.5),
            MoveAlongPath(bike_straight, straight_line, rate_func=lambda t: t**2, run_time=2.5), 
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)

        # Graph Velocity vs Time
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 5, 1],
            x_length=4,
            y_length=2,
            axis_config={"include_tip": True, "font_size": 14},
            tips=False
        )
        # Position axes according to Critic fix (Issue 33)
        self.place_in_area(axes, 'E4', 'F6', scale_factor=0.6)
        
        labels = axes.get_axis_labels(
            x_label=Text("Time", font_size=14), 
            y_label=Text("Velocity", font_size=14)
        )
        
        graph_straight = axes.plot(lambda x: 0.8 * x, x_range=[0, 4], color=straight_color)
        graph_curve = axes.plot(lambda x: 4 * (1 - np.exp(-1.2 * x)), x_range=[0, 4], color=curve_color)
        
        self.play(Create(axes), FadeIn(labels))
        self.play(Create(graph_straight), Create(graph_curve))
        self.wait(2)

        # Clean up highlights
        self.lecture[4].set_color(WHITE)
        self.wait(2)
