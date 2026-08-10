from manim import *

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
        self.setup_layout("Shrinking the Interval (The Secant Line)", [
            "Connect two points with a secant line.",
            "Watch the interval shrink on our graph.",
            "The secant line moves closer to the curve."
        ])
        
        # Define the curve (Asset usage)
        curve_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/curve.svg")
        axes = Axes(x_range=[0, 6, 1], y_range=[0, 6, 1], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: 0.2 * x**2, x_range=[0, 5], color=YELLOW)
        
        area = VGroup(axes, curve, curve_svg)
        self.place_in_area(area, 'A1', 'D2', scale_factor=0.65) # Adjusted per Issue 25
        
        # Points
        p1 = Dot(color=BLUE)
        p2 = Dot(color=RED)
        p1.move_to(axes.c2p(1, 0.2))
        p2.move_to(axes.c2p(4, 3.2))
        
        # Secant line
        secant = Line(p1.get_center(), p2.get_center(), color=WHITE)
        secant_line_group = VGroup(secant, p1, p2)
        
        # Slope value
        slope_val = MathTex(r"\\text{slope} = 1.0", font_size=30, color=WHITE)
        self.place_at_grid(slope_val, 'A4', scale_factor=0.8) # Adjusted per Issue 24/37

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(FadeIn(p1), FadeIn(p2), Create(secant), Write(slope_val))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(RED))
        
        # Shrink interval
        for i in range(10):
            t2 = 4 - i * 0.3
            p2_new = axes.c2p(t2, 0.2 * t2**2)
            
            new_secant = Line(p1.get_center(), p2_new, color=WHITE)
            self.play(
                p2.animate.move_to(p2_new),
                Transform(secant, new_secant),
                run_time=0.2
            )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color("#00FFFF"))
        self.play(slope_val.animate.set_color("#00FFFF"))
        self.wait(2)
