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
        # Setup title and lecture lines from storyboard
        title_str = "Prerequisite: The Concept of Slope"
        lecture_lines_str = [
            "- Slope measures how steep a straight line is.",
            "- We calculate it as vertical rise over horizontal run.",
            "- On a curve, the steepness changes at every point."
        ]
        self.setup_layout(title_str, lecture_lines_str)
        
        # === Animation for Lecture Line 1 ===
        # Create a straight white line (#FFFFFF) and a simple ant shape on it.
        self.lecture[0].set_color(WHITE)
        
        start_pt = self.grid["D1"]
        end_pt = self.grid["B6"]
        white_line = Line(start_pt, end_pt, color=WHITE)
        
        # Simple ant shape (Mobject group)
        ant_body = Ellipse(width=0.4, height=0.2, color=WHITE, fill_opacity=1)
        ant_head = Dot(radius=0.08, color=WHITE).next_to(ant_body, RIGHT, buff=0)
        ant = VGroup(ant_body, ant_head)
        ant.move_to(start_pt)
        # Initial orientation along the line
        init_angle = white_line.get_angle()
        ant.rotate(init_angle)
        ant.current_angle = init_angle

        self.play(Create(white_line))
        self.play(FadeIn(ant))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight 'rise' and 'run' with a green triangle (#00FF00).
        self.lecture[1].set_color(GREEN)
        
        # Define triangle points using grid-aligned positions
        p_corner = np.array([end_pt[0], start_pt[1], 0])
        triangle = Polygon(start_pt, p_corner, end_pt, color=GREEN, fill_opacity=0.3, stroke_width=2)
        
        # Fix Issue 27: Reposition 'run' label to E3
        run_label = Text("run", color=GREEN, font_size=20)
        self.place_at_grid(run_label, 'E3', scale_factor=0.6)
        
        # Fix Issue 28: Reposition 'rise' label to D6
        rise_label = Text("rise", color=GREEN, font_size=20)
        self.place_at_grid(rise_label, 'D6', scale_factor=0.6)
        
        self.play(Create(triangle), Write(run_label), Write(rise_label))
        
        # Move ant along the plank (line)
        self.play(ant.animate.move_to(end_pt), run_time=2, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition to a blue curve (#0000FF). Ant walks with a white tangent line (#FFFFFF).
        self.lecture[2].set_color(BLUE)
        
        # Define a blue curve
        curve = ParametricFunction(
            lambda t: np.array([t, 0.5 * np.sin(t), 0]),
            t_range=[-3, 3],
            color=BLUE
        )
        # Fix Issue 29: Position curve in area B2 to E6
        self.place_in_area(curve, 'B2', 'E6', scale_factor=0.8)
        
        # White tangent line that follows the ant
        tangent_line = Line(LEFT, RIGHT, color=WHITE).scale(0.7)
        
        # Transition sequence
        self.play(
            FadeOut(white_line),
            FadeOut(triangle),
            FadeOut(run_label),
            FadeOut(rise_label),
            Create(curve),
            # Orient ant for the start of the curve (approx 0 angle for sin(t) at -3)
            ant.animate.move_to(curve.point_from_proportion(0)).rotate(-ant.current_angle)
        )
        ant.current_angle = 0
        
        # Movement along curve using ValueTracker and updaters
        t_tracker = ValueTracker(0)
        
        def ant_updater(m):
            t = t_tracker.get_value()
            pos = curve.point_from_proportion(t)
            # Numerical tangent calculation for rotation
            dt = 0.001
            p1 = curve.point_from_proportion(max(0, t - dt))
            p2 = curve.point_from_proportion(min(1, t + dt))
            target_angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
            m.move_to(pos)
            m.rotate(target_angle - m.current_angle)
            m.current_angle = target_angle
            
        def tangent_updater(m):
            m.move_to(ant.get_center())
            m.set_angle(ant.current_angle)
            
        ant.add_updater(ant_updater)
        tangent_line.add_updater(tangent_updater)
        
        self.add(tangent_line)
        self.play(t_tracker.animate.set_value(1), run_time=5, rate_func=linear)
        
        # Remove updaters and clean up for final markers
        ant.remove_updater(ant_updater)
        tangent_line.remove_updater(tangent_updater)
        self.play(FadeOut(ant), FadeOut(tangent_line))
        
        # Highlight slope at three distinct points: yellow (#FFFF00), orange (#FFA500), red (#FF0000)
        marker_proportions = [0.2, 0.5, 0.8]
        marker_colors = [YELLOW, "#FFA500", RED]
        
        for p_prop, m_color in zip(marker_proportions, marker_colors):
            pos = curve.point_from_proportion(p_prop)
            # Calculate tangent angle at this point
            dt = 0.001
            p1 = curve.point_from_proportion(p_prop - dt)
            p2 = curve.point_from_proportion(p_prop + dt)
            angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
            
            dot = Dot(pos, color=m_color)
            line = Line(LEFT, RIGHT, color=m_color).scale(0.6).move_to(pos).set_angle(angle)
            
            self.play(FadeIn(dot), Create(line), run_time=1)
            
        self.wait(2)
