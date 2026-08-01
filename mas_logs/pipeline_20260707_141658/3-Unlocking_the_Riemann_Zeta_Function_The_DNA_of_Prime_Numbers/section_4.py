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
        # Lecture lines from storyboard
        lecture_lines = [
            "What if \"s\" is a complex coordinate?",
            "We map \"s\" to a point on the complex plane.",
            "The Zeta function transforms this plane into swirling maps.",
            "This reveals hidden structures beyond simple real numbers.",
            "We use analytic continuation to extend the function's reach."
        ]
        
        self.setup_layout("Entering the Complex Plane", lecture_lines)

        # Colors
        COLOR_GRID = "#808080"
        COLOR_GLOW_GREEN = "#00FF00"
        COLOR_MORPH = "#FFD700"
        COLOR_GLOW_RED = "#FF0000"
        COLOR_TEXT = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_GRID))
        
        # Build a grid of lines
        plane_grid = VGroup()
        for i in range(-3, 4):
            # Horizontal lines
            h_line = Line(3*LEFT, 3*RIGHT, stroke_width=1, color=COLOR_GRID).move_to(i * 0.8 * UP)
            plane_grid.add(h_line)
            # Vertical lines
            v_line = Line(3*DOWN, 3*UP, stroke_width=1, color=COLOR_GRID).move_to(i * 0.8 * RIGHT)
            plane_grid.add(v_line)
        
        # Updated positioning based on Issue 31
        self.place_in_area(plane_grid, 'B2', 'F6', scale_factor=0.6)
        
        # Updated positioning based on Issue 33
        s_label = Text("s = σ + it", font_size=36, color=COLOR_TEXT)
        self.place_at_grid(s_label, 'A4', scale_factor=0.8)
        
        self.play(Create(plane_grid), Write(s_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_GLOW_GREEN))
        
        highlight_green = Rectangle(
            width=2.0, height=3.5, 
            fill_color=COLOR_GLOW_GREEN, fill_opacity=0.2, 
            stroke_width=0
        )
        highlight_green.move_to(plane_grid.get_center() + 1.2 * RIGHT)
        
        # Updated positioning based on Issue 33
        re_label = Text("Re(s) > 1", font_size=24, color=COLOR_GLOW_GREEN)
        self.place_at_grid(re_label, 'B6', scale_factor=0.8)
        
        self.play(FadeIn(highlight_green), Write(re_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_MORPH))
        
        def swirl_func(point):
            x, y, z = point
            cx, cy, cz = plane_grid.get_center()
            rx, ry = x - cx, y - cy
            dist = np.sqrt(rx**2 + ry**2) + 0.1
            angle = np.arctan2(ry, rx) + 1.5 / dist
            new_x = dist * np.cos(angle)
            new_y = dist * np.sin(angle)
            return np.array([new_x + cx, new_y + cy, 0])

        self.play(
            plane_grid.animate.apply_function(swirl_func).set_color(COLOR_MORPH),
            highlight_green.animate.apply_function(swirl_func).set_color(COLOR_MORPH).set_opacity(0.1),
            run_time=3
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(COLOR_GLOW_RED))
        
        highlight_red = Rectangle(
            width=2.5, height=3.5, 
            fill_color=COLOR_GLOW_RED, fill_opacity=0.2, 
            stroke_width=0
        )
        highlight_red.move_to(plane_grid.get_center() + 1.2 * LEFT)
        
        # Updated positioning based on Issue 31
        re_less_label = Text("Re(s) < 1", font_size=24, color=COLOR_GLOW_RED)
        self.place_at_grid(re_less_label, 'B3', scale_factor=0.8)
        
        self.play(FadeIn(highlight_red), Write(re_less_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(COLOR_TEXT))
        
        ac_text = Text("Analytic Continuation", font_size=28, color=COLOR_TEXT)
        # Updated positioning based on Issue 32
        self.place_in_area(ac_text, 'E2', 'E4', scale_factor=0.6)
        
        self.play(FadeIn(ac_text))
        self.wait(3)
