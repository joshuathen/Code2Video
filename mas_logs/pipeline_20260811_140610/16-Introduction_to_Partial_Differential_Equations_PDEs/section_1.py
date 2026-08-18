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
            "ODEs track change in one variable, like time.",
            "PDEs track change across both space and time.",
            "Think of a point moving on a line.",
            "Contrast this with ripples across a pond's surface.",
            "PDEs capture this complex spatial evolution."
        ]
        self.setup_layout("From ODEs to PDEs", lecture_lines)
        
        # Assets
        point_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/point.svg")
        line_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/line.svg")
        surface_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/surface.svg")
        
        # === Animation for Lecture Line 1 ===
        # ODE: x(t)
        ode_eq = MathTex(r"x(t)", color="#4287F5")
        self.place_at_grid(ode_eq, 'C2', scale_factor=0.8)
        self.play(Write(ode_eq))
        self.play(self.lecture[0].animate.set_color("#4287F5"))

        # === Animation for Lecture Line 2 ===
        # PDE: u(x,t)
        pde_eq = MathTex(r"u(x, t)", color="#FF9F00")
        self.place_at_grid(pde_eq, 'C5', scale_factor=0.8)
        self.play(Write(pde_eq))
        self.play(self.lecture[1].animate.set_color("#FF9F00"))

        # === Animation for Lecture Line 3 ===
        # Use assets point.svg and line.svg
        self.place_at_grid(line_asset, "D2", scale_factor=0.6)
        self.place_at_grid(point_asset, "D2", scale_factor=0.4)
        self.add(line_asset, point_asset)
        self.play(point_asset.animate.shift(RIGHT * 1), run_time=2)
        self.play(self.lecture[2].animate.set_color(YELLOW))

        # === Animation for Lecture Line 4 ===
        # Contrast with ripples on surface (pond)
        self.place_at_grid(surface_asset, "D5", scale_factor=0.5)
        pond = Circle(radius=0.8, color=BLUE).set_fill(BLUE, opacity=0.3)
        self.place_in_area(pond, 'D4', 'E6', scale_factor=0.9)
        ripple1 = Circle(radius=0.2, color=BLUE)
        ripple2 = Circle(radius=0.4, color=BLUE)
        self.place_at_grid(ripple1, 'D5', scale_factor=0.8)
        self.place_at_grid(ripple2, 'D5', scale_factor=0.8)
        self.add(surface_asset, pond, ripple1, ripple2)
        self.play(
            ripple1.animate.scale(2), 
            ripple2.animate.scale(1.5),
            run_time=2
        )
        self.play(self.lecture[3].animate.set_color(BLUE))

        # === Animation for Lecture Line 5 ===
        final_box = SurroundingRectangle(VGroup(ode_eq, pde_eq), color=WHITE, buff=0.5)
        self.play(Create(final_box))
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(2)
