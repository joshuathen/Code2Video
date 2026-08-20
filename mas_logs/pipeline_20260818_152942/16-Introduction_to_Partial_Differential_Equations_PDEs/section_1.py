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
        self.setup_layout("From ODEs to PDEs: Multi-Dimensional Change", 
                         ["ODEs track change in one dimension.", 
                          "PDEs track change in multiple dimensions.", 
                          "Visualize a rolling ball versus drumhead."])
        
        # === Animation for Lecture Line 1 ===
        # ODE: 1D path with ball asset
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 3, 1], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: 0.5 * x**2, x_range=[0, 3.5], color=BLUE)
        ball1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg", color=RED)
        
        self.place_in_area(VGroup(axes, curve, ball1), 'A1', 'C6', scale_factor=0.6)
        self.play(FadeIn(axes), Create(curve), FadeIn(ball1))
        self.play(self.lecture[0].animate.set_color(BLUE))

        # === Animation for Lecture Line 2 ===
        # PDE: 3D surface
        axes3d = ThreeDAxes(x_range=[-2, 2, 1], y_range=[-2, 2, 1], z_range=[-1, 1, 1])
        surface = Surface(
            lambda u, v: np.array([u, v, 0.5 * np.sin(u) * np.cos(v)]),
            u_range=[-2, 2], v_range=[-2, 2],
            fill_opacity=0.8, fill_color=YELLOW
        )
        
        self.play(FadeOut(axes), FadeOut(curve), FadeOut(ball1))
        # Updated positioning per issue 21, 22, 36
        self.place_in_area(axes3d, 'D1', 'F3', scale_factor=0.3)
        self.place_in_area(surface, 'D4', 'F6', scale_factor=0.3)
        self.play(FadeIn(axes3d), Create(surface))
        self.play(self.lecture[1].animate.set_color(YELLOW))

        # === Animation for Lecture Line 3 ===
        # Drumhead visual with ball asset
        drumhead = Circle(radius=1.5, color=GREEN, fill_opacity=0.5)
        ball2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg", color=RED)
        
        # Updated positioning per issue 23, 36
        self.place_at_grid(drumhead, 'B5', scale_factor=0.7)
        self.place_at_grid(ball2, 'B5', scale_factor=0.3)
        
        self.play(FadeOut(axes3d), FadeOut(surface))
        self.play(Create(drumhead), FadeIn(ball2))
        self.play(self.lecture[2].animate.set_color(GREEN))
        self.wait(1)
