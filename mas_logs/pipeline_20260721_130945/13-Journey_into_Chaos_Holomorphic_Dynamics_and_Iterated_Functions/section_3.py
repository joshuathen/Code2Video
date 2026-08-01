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
        # Data from storyboard
        title_text = "Orbits: The Path of a Point"
        lecture_lines = [
            "An orbit is the sequence of points generated.",
            "Some points spiral inward toward a stable attractor.",
            "Others escape, shooting off to infinity very quickly.",
            "Some may wander forever in a repeating cycle.",
            "Every point follows a unique path through the plane."
        ]
        
        # Colors for matching elements to lecture lines
        colors = [WHITE, "#00FF00", "#FF4500", "#FFFF00", "#00FFFF"]
        
        # 1. Initialize Layout
        self.setup_layout(title_text, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1
        self.play(self.lecture[0].animate.set_color(colors[0]))
        
        # Formula z_{n+1} = z_n^2 + c in #FFFFFF at the top.
        # Per Issue 28: use place_in_area for formula at A3-A4
        self.func_label = MathTex("z_{n+1} = z_n^2 + c", color=colors[0])
        self.place_in_area(self.func_label, 'A3', 'A4', scale_factor=0.9)
        
        # Create a plane to visualize orbits
        # Plane is placed in the central-right area
        self.plane = ComplexPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=4,
            y_length=4,
            background_line_style={"stroke_opacity": 0.3}
        )
        self.place_in_area(self.plane, 'B2', 'E5')
        
        self.play(
            Write(self.func_label),
            Create(self.plane),
            run_time=1.5
        )
        
        # === Animation for Lecture Line 2 ===
        # Highlight lecture line 2
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(colors[1])
        )
        
        # Place P1 at (0.1, 0.1) in #00FF00
        z1_start = 0.5 + 0.5j # Larger start for better visibility of spiral
        p1 = Dot(self.plane.n2p(z1_start), color=colors[1])
        p1_label = Text("P1", font_size=18, color=colors[1]).next_to(p1, UR, buff=0.1)
        self.play(FadeIn(p1), Write(p1_label))
        
        # Animate P1 moving in a small, bounded spiral near the origin.
        # We iterate z^2 where |z|<1 so it spirals into the attractor at 0.
        path_points = [z1_start]
        curr_z = z1_start
        for _ in range(6):
            curr_z = curr_z**2
            path_points.append(curr_z)
        
        path_p1 = VMobject(color=colors[1], stroke_width=2)
        path_p1.set_points_as_corners([self.plane.n2p(z) for z in path_points])
        
        self.play(
            MoveAlongPath(p1, path_p1),
            p1_label.animate.move_to(self.plane.n2p(path_points[-1]) + UR*0.1),
            run_time=2
        )
        
        # === Animation for Lecture Line 3 ===
        # Highlight lecture line 3
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(colors[2])
        )
        
        # Place P2 at (1.5, 1.5) in #FF4500
        z2_start = 1.5 + 1.5j
        p2 = Dot(self.plane.n2p(z2_start), color=colors[2])
        p2_label = Text("P2", font_size=18, color=colors[2]).next_to(p2, UR, buff=0.1)
        self.play(FadeIn(p2), Write(p2_label))
        
        # Animate P2 shooting off-screen rapidly as its values grow.
        z2_end = (z2_start**2) * 2 # Point far outside the visible area
        
        self.play(
            p2.animate.move_to(self.plane.n2p(z2_end)),
            p2_label.animate.move_to(self.plane.n2p(z2_end) + UR*0.1),
            run_time=1.5,
            rate_func=rate_functions.ease_in_cubic
        )
        
        # === Animation for Lecture Line 4 ===
        # Highlight lecture line 4
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(colors[3])
        )
        
        # Per Issue 29: use place_in_area for chaos label at F3-F4
        self.chaos_text = Text("Cycle & Chaos", font_size=22, color=colors[3])
        self.place_in_area(self.chaos_text, 'F3', 'F4', scale_factor=0.8)
        self.play(Write(self.chaos_text))
        
        # === Animation for Lecture Line 5 ===
        # Highlight lecture line 5
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(colors[4])
        )
        
        # Circle the bounded path of P1 with a soft glow in #00FF00.
        glow_circle = Circle(radius=0.4, color=colors[1], stroke_width=4).move_to(self.plane.n2p(0))
        glow_circle.set_fill(colors[1], opacity=0.3)
        
        self.play(FadeIn(glow_circle))
        # Per L004: use Indicate() for highlighting
        self.play(Indicate(glow_circle, color=colors[1], scale_factor=1.2))
        
        self.wait(2)
