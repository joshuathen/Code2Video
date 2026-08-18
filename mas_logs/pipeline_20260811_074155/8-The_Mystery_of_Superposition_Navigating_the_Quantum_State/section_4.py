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

class Section4Scene(TeachingScene):
    def construct(self):
        # Lecture lines and Title
        title = "Visualizing the Math: The Bloch Sphere"
        lines = [
            "We can map quantum states onto a 3D sphere.",
            "The North Pole represents the pure state zero.",
            "The South Pole represents the pure state one.",
            "Any point on the surface is a unique superposition.",
            "The equator marks a perfect fifty-fifty mix of states."
        ]
        
        self.setup_layout(title, lines)
        
        # Colors for matching lecture lines
        c_sphere = "#CCCCCC"
        c_north = "#00FF00"
        c_south = "#FF00FF"
        c_point = "#0000FF"
        c_equator = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # Map quantum states onto a 3D sphere.
        self.lecture[0].set_color(c_sphere)
        
        # Create a wireframe sphere representation
        sphere_outline = Circle(radius=2.0, color=c_sphere, stroke_width=2)
        equator_wire = Ellipse(width=4.0, height=0.8, color=c_sphere, stroke_width=1).set_stroke(opacity=0.5)
        meridian_wire = Ellipse(width=0.8, height=4.0, color=c_sphere, stroke_width=1).set_stroke(opacity=0.5)
        
        sphere_group = VGroup(sphere_outline, equator_wire, meridian_wire)
        # Apply layout fix: scale to 0.8 to prevent cramping
        self.place_in_area(sphere_group, "A1", "F6", scale_factor=0.8)
        
        self.play(Create(sphere_outline), Create(equator_wire), Create(meridian_wire))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The North Pole represents the pure state zero.
        self.lecture[1].set_color(c_north)
        
        north_pole_pos = sphere_outline.get_top()
        north_dot = Dot(north_pole_pos, color=c_north)
        north_label = MathTex(r"|0\rangle", color=c_north, font_size=24)
        north_label.next_to(north_dot, UP, buff=0.1)
        
        self.play(FadeIn(north_dot), Write(north_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The South Pole represents the pure state one.
        self.lecture[2].set_color(c_south)
        
        south_pole_pos = sphere_outline.get_bottom()
        south_dot = Dot(south_pole_pos, color=c_south)
        south_label = MathTex(r"|1\rangle", color=c_south, font_size=24)
        south_label.next_to(south_dot, DOWN, buff=0.1)
        
        self.play(FadeIn(south_dot), Write(south_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Any point on the surface is a unique superposition.
        self.lecture[3].set_color(c_point)
        
        # Define a point on the surface (visual approximation of surface location)
        center = sphere_group.get_center()
        surface_point_pos = center + np.array([1.2 * 0.8, 0.8 * 0.8, 0])
        surface_dot = Dot(surface_point_pos, color=c_point)
        
        # Pulse animation from equator to surface point
        pulse_start = center + np.array([1.6, 0, 0])
        pulse = Dot(pulse_start, color=c_point, radius=0.05)
        
        self.play(FadeIn(pulse))
        self.play(pulse.animate.move_to(surface_point_pos), run_time=1.5)
        self.play(pulse.animate.scale(2).set_opacity(0), FadeIn(surface_dot))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The equator marks a perfect fifty-fifty mix of states.
        self.lecture[4].set_color(c_equator)
        
        equator_ring = Ellipse(width=4.0 * 0.8, height=0.8 * 0.8, color=c_equator, stroke_width=4)
        equator_ring.move_to(center)
        
        # Glowing effect simulation
        glow = equator_ring.copy().set_stroke(width=10, opacity=0.3)
        
        self.play(Create(equator_ring), FadeIn(glow))
        self.play(Indicate(equator_ring, color=c_equator))
        self.wait(2)
