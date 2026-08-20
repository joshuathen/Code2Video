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
        self.setup_layout("The Dynamic Transition: Zooming In", [
            "Zooming in reveals the microscopic view.", 
            "Curves become lines when zoomed close enough.", 
            "Derivatives define this local linear behavior."
        ])
        
        # Define objects
        axes = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"color": WHITE}).scale(0.5)
        curve = axes.plot(lambda x: 0.5 * x**3 - x, color=YELLOW)
        point = Dot(axes.c2p(1, -0.5), color=RED)
        tangent = TangentLine(curve, alpha=0.75, length=1, color=GREEN)
        
        # Assets
        magnifier = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifying.svg").scale(0.3)
        microscope = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microscope.svg").scale(0.3)
        
        # Position Assets
        self.place_at_grid(magnifier, "A6")
        self.place_at_grid(microscope, "F6")

        # Position main objects - quadrant D4-F6 (Right-hand lower area)
        self.place_in_area(axes, "D4", "F6", scale_factor=0.6)
        # Re-plot curve inside the moved axes context or shift it
        curve.move_to(axes.get_center())
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(magnifier), Create(axes), Create(curve), FadeIn(point))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.play(
            axes.animate.scale(2),
            curve.animate.scale(2),
            run_time=2
        )
        self.play(Create(tangent))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        formula = MathTex(r"f'(a) = \lim_{h \to 0} \frac{f(a+h)-f(a)}{h}", color=WHITE)
        # Positioned in E3 as requested
        self.place_at_grid(formula, "E3", scale_factor=0.5)
        self.play(self.lecture[2].animate.set_color(BLUE))
        self.play(FadeIn(microscope), Write(formula))
        self.wait(2)
