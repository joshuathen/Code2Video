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
        self.setup_layout("Summary and Intuition", [
            "Phase space paths trace a circular arc.",
            "Counting collisions measures the arc's length.",
            "Physics naturally emerges from the geometry of Pi."
        ])
        
        # === Assets ===
        billiard_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/billiard.svg")
        ball_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg")
        
        # === Animation for Lecture Line 1 ===
        # Summarize collision events as a series of reflections using [Asset: .../billiard.svg], #FF0000.
        circle = Circle(radius=1.2, color=BLUE)
        self.place_in_area(circle, "C4", "E6")
        
        arc = Arc(radius=1.2, start_angle=0, angle=PI/2, color="#FF0000", stroke_width=6)
        arc.move_to(circle.get_center())
        
        billiard_icon = billiard_svg.copy()
        self.place_at_grid(billiard_icon, "B5", scale_factor=0.6)
        
        self.add(circle)
        self.play(Create(arc), FadeIn(billiard_icon))
        self.lecture[0].set_color("#FF0000")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate total reflections of [Asset: .../ball.svg] converging to value of Pi, #00FF00.
        dots = VGroup(*[Dot(point=arc.point_from_proportion(i/10), color="#00FF00") for i in range(11)])
        
        ball_img = ball_icon.copy()
        self.place_at_grid(ball_img, "D6", scale_factor=0.6)
        
        self.play(Create(dots), FadeIn(ball_img))
        self.lecture[1].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Fade out all elements leaving only the Pi symbol, #FFFFFF.
        pi_sym = MathTex(r"\\pi", font_size=144, color=WHITE)
        self.place_in_area(pi_sym, "C4", "E6", scale_factor=0.5)
        
        self.play(
            FadeOut(circle),
            FadeOut(arc),
            FadeOut(dots),
            FadeOut(billiard_icon),
            FadeOut(ball_img),
            FadeIn(pi_sym)
        )
        self.lecture[2].set_color("#FFFFFF")
        self.wait(2)
