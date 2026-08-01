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
        # Initialize lecture lines
        lecture_lines = [
            "Since the state is finite, the process is periodic.",
            "The windmill returns to its start and repeats forever.",
            "This beautiful cycle connects every point in the set."
        ]
        self.setup_layout("Summary and the Eternal Cycle", lecture_lines)

        # 1. Prepare points (point_set)
        # Fix for Issue 36: point_set anchored to grid area B2 to E5
        point_coords = [
            [-1.2, 0.8, 0], [1.1, 1.0, 0], [0.6, -0.4, 0],
            [-0.9, -0.7, 0], [1.3, -0.9, 0], [-0.1, 0.3, 0]
        ]
        point_set = VGroup(*[Dot(pos, color=WHITE) for pos in point_coords])
        self.place_in_area(point_set, 'B2', 'E5', scale_factor=0.8)
        self.add(point_set)

        # 2. Prepare windmill asset and line
        # Issue 23: Asset Integration
        windmill_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/windmill.svg")
        windmill_asset.set_color(YELLOW).scale(0.3)
        
        # Initial pivot is the first point in the set
        pivot_point = point_set[0].get_center()
        windmill_line = Line(start=pivot_point + 2.2*LEFT, end=pivot_point + 2.2*RIGHT, color=YELLOW, stroke_width=4)
        initial_angle = 30 * DEGREES
        windmill_line.rotate(initial_angle, about_point=pivot_point)
        windmill_asset.move_to(pivot_point)
        
        self.add(windmill_line, windmill_asset)

        # === Animation for Lecture Line 1 ===
        # "Since the state is finite, the process is periodic."
        self.play(self.lecture[0].animate.set_color(YELLOW))

        # Fast-forward rotation, leaving thin gray (#808080) traces
        num_traces = 35
        windmill_traces = VGroup()
        for i in range(num_traces):
            ang = initial_angle + i * (5 * PI / num_traces)
            # Pick points from the set to pivot around in a simulated cycle
            p_idx = (i // 6) % len(point_set)
            p = point_set[p_idx].get_center()
            t = Line(p + 2.5*LEFT, p + 2.5*RIGHT, color="#808080", stroke_width=1).set_opacity(0.3)
            t.rotate(ang, about_point=p)
            windmill_traces.add(t)

        # Fix for Issue 35: windmill_traces restricted to A2 to F6 to avoid obstruction
        self.place_in_area(windmill_traces, 'A2', 'F6', scale_factor=0.6)
        
        self.play(Create(windmill_traces, lag_ratio=0.04), run_time=2.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "The windmill returns to its start and repeats forever."
        self.play(
            self.lecture[1].animate.set_color(YELLOW),
            self.lecture[0].animate.set_color(WHITE)
        )
        
        # Return to start state: Fade out the web of traces and return to initial pivot
        self.play(
            FadeOut(windmill_traces),
            windmill_line.animate.set_color(YELLOW),
            run_time=1.0
        )
        
        # Pulse animation at the starting point
        self.play(
            windmill_line.animate.scale(1.2), 
            windmill_asset.animate.scale(1.2), 
            run_time=0.25
        )
        self.play(
            windmill_line.animate.scale(1/1.2), 
            windmill_asset.animate.scale(1/1.2), 
            run_time=0.25
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "This beautiful cycle connects every point in the set."
        self.play(
            self.lecture[2].animate.set_color(YELLOW),
            self.lecture[1].animate.set_color(WHITE)
        )

        # Fix for Issue 37: highlight_dots synchronized with the point set's grid area
        highlight_dots = VGroup(*[Dot(p.get_center(), color="#2ECC71") for p in point_set])
        self.place_in_area(highlight_dots, 'B2', 'E5', scale_factor=0.8)
        
        # Visual highlight of the points
        self.play(
            Flash(point_set, color="#2ECC71", line_length=0.3, num_lines=15),
            FadeIn(highlight_dots),
            point_set.animate.set_opacity(0),
            run_time=1
        )
        
        # Repetitive pulse to signify the "eternal cycle"
        for _ in range(2):
            self.play(highlight_dots.animate.scale(1.3), run_time=0.4)
            self.play(highlight_dots.animate.scale(1/1.3), run_time=0.4)
            
        self.wait(2)
