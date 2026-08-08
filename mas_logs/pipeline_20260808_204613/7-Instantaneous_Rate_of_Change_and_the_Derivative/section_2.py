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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Visualizing the 'Shrinking' Interval", [
            "Two points define a secant line.",
            "As the points get closer, the gap shrinks.",
            "The secant line tilts as the gap closes.",
            "At the limit, it touches one point.",
            "This is the tangent line."
        ])

        # Setup Axes and Graph
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: 0.5 * x**2, color=BLUE)
        graph = VGroup(axes, curve)
        
        # Initial placement per constraint
        self.place_in_area(graph, 'B3', 'F6', scale_factor=0.8)

        a_val = 2.0
        h = ValueTracker(1.0)
        
        # Create persistent mobjects
        point_a = Dot(axes.c2p(a_val, 0.5 * a_val**2), color=YELLOW)
        point_b = Dot(color=RED)
        secant = Line(color=GREEN)
        
        # Use updaters for smooth animation
        def update_point_b(mob):
            mob.move_to(axes.c2p(a_val + h.get_value(), 0.5 * (a_val + h.get_value())**2))
            
        def update_secant(mob):
            p1 = axes.c2p(a_val, 0.5 * a_val**2)
            p2 = axes.c2p(a_val + h.get_value(), 0.5 * (a_val + h.get_value())**2)
            mob.become(Line(p1, p2, color=GREEN))
            
        point_b.add_updater(update_point_b)
        secant.add_updater(update_secant)
        
        # Initial update to place correctly
        update_point_b(point_b)
        update_secant(secant)
        
        self.add(graph, point_a, point_b, secant)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF0000"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        self.play(h.animate.set_value(0.5), run_time=1.5)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        self.play(h.animate.set_value(0.1), run_time=1.5)
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#00FFFF"))
        self.play(h.animate.set_value(0.01), run_time=1.5)
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF00FF"))
        self.wait(2)
