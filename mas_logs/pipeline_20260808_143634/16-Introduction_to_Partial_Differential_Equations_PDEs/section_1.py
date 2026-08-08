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
        lecture_lines = ["ODEs model change over time.", "PDEs model change over space and time.", "A pendulum is a simple ODE.", "A pond ripple is a PDE.", "They map points to surfaces."]
        self.setup_layout("From ODEs to PDEs: A Visual Shift", lecture_lines)
        
        # Pre-build objects
        ode_label = Text("ODE", color=YELLOW)
        pde_label = Text("PDE", color=BLUE)
        
        point = Dot(color=YELLOW)
        time_axis = Line(LEFT*2, RIGHT*2, color=WHITE)
        
        # Asset loading
        pendulum_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pendulum.svg")
        pond_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pond.svg")
        
        surface = Surface(
            lambda u, v: np.array([u, v, 0.5 * np.sin(u) * np.cos(v)]),
            u_range=[-2, 2], v_range=[-2, 2],
            resolution=(15, 15)
        ).set_style(fill_opacity=0.5, fill_color=BLUE, stroke_width=0.5)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.place_at_grid(time_axis, 'C3')
        # Fix 23: adjusted point placement
        self.place_at_grid(point, 'D3', scale_factor=0.8)
        self.play(FadeIn(time_axis), FadeIn(point))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        # Fix 22: adjusted surface placement
        self.place_in_area(surface, 'C4', 'F6', scale_factor=0.6)
        self.play(FadeIn(surface))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        # Asset integration
        self.place_at_grid(pendulum_icon, 'B3', scale_factor=0.5)
        # Fix 21: adjusted label placement
        self.place_at_grid(ode_label, 'B2', scale_factor=0.8)
        self.play(Write(ode_label), FadeIn(pendulum_icon))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(BLUE)
        # Asset integration
        self.place_at_grid(pond_icon, 'B6', scale_factor=0.5)
        # Fix 21: adjusted label placement
        self.place_at_grid(pde_label, 'B5', scale_factor=0.8)
        self.play(Write(pde_label), FadeIn(pond_icon))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(PURPLE)
        self.play(point.animate.move_to(self.grid['F3']), surface.animate.set_color(PURPLE))
        self.wait(1)
