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
        # Setup with mandatory lines
        title = "The Mechanism: The 'Shopping Cart' Analogy"
        lines = [
            'A shopping cart moves smoothly across the paved sidewalk.',
            'One front wheel hits the grass and slows down.',
            'This imbalance causes the entire cart to pivot.',
            'The cart now travels at a sharper new angle.',
            'Light waves bend exactly like this angled shopping cart.'
        ]
        self.setup_layout(title, lines)

        # Environment setup: Pavement (Rows A-C) and Grass (Rows D-F)
        pavement = Rectangle(width=6, height=3, fill_color="#AAAAAA", fill_opacity=0.3, stroke_width=0)
        grass = Rectangle(width=6, height=3, fill_color="#228B22", fill_opacity=0.4, stroke_width=0)
        self.place_in_area(pavement, 'A1', 'C6')
        self.place_in_area(grass, 'D1', 'F6')
        
        # Boundary line (between row C and D, roughly y = -0.3)
        boundary = Line(self.grid['C1'] + 0.5*DOWN, self.grid['C6'] + 0.5*DOWN, color=WHITE, stroke_opacity=0.5)
        self.add(pavement, grass, boundary)

        # Cart setup
        cart_body = Rectangle(width=0.8, height=1.2, fill_color="#808080", fill_opacity=1, stroke_width=1, stroke_color=WHITE)
        w_fl = Circle(radius=0.1, fill_color="#333333", fill_opacity=1, stroke_width=0).shift(0.3 * LEFT + 0.4 * UP)
        w_fr = Circle(radius=0.1, fill_color="#333333", fill_opacity=1, stroke_width=0).shift(0.3 * RIGHT + 0.4 * UP)
        w_bl = Circle(radius=0.1, fill_color="#333333", fill_opacity=1, stroke_width=0).shift(0.3 * LEFT + 0.4 * DOWN)
        w_br = Circle(radius=0.1, fill_color="#333333", fill_opacity=1, stroke_width=0).shift(0.3 * RIGHT + 0.4 * DOWN)
        cart = VGroup(cart_body, w_fl, w_fr, w_bl, w_br).rotate(-20 * DEGREES)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        # Resolved Issue 40/41/56: Position at B4, scale 0.6
        self.place_at_grid(cart, 'B4', scale_factor=0.6)
        self.play(FadeIn(cart))
        self.play(cart.animate.shift(DOWN * 0.8 + RIGHT * 0.3), run_time=1.5)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        # Move until the front-right wheel hits the boundary
        # The wheel color changes to red to indicate friction/slowing down
        self.play(
            cart.animate.shift(DOWN * 0.4 + RIGHT * 0.15),
            w_fr.animate.set_color("#FF0000"),
            run_time=1
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Pivot the cart around the stuck wheel (w_fr)
        pivot_point = w_fr.get_center()
        self.play(
            Rotate(cart, angle=-25 * DEGREES, about_point=pivot_point),
            run_time=1.5
        )
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Move the cart forward at the new angle
        # Heading direction is now deeper into the grass
        self.play(cart.animate.shift(DOWN * 1.0 + RIGHT * 0.1), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        self.play(FadeOut(cart))
        
        # Wavefronts: Horizontal-ish lines moving across the boundary
        # Resolved Issue 39/40/56: Position B4, scale 0.7 to avoid clipping and text crowding
        wavefronts = VGroup(*[
            Line(LEFT*1.8, RIGHT*1.8, color=WHITE, stroke_width=2).shift(UP*i*0.5).rotate(-20*DEGREES)
            for i in range(4)
        ])
        self.place_at_grid(wavefronts, 'B4', scale_factor=0.7)
        
        # Prepare bent wavefronts to represent refraction
        bent_wavefronts = VGroup()
        for i in range(4):
            # Intersection point at boundary y=-0.3
            # We construct each bent wave as two segments
            top_seg = Line(self.grid['B4'] + UP*i*0.5 + LEFT*1.2, self.grid['C4'] + UP*i*0.5 + LEFT*0.2, color=WHITE).rotate(-20*DEGREES)
            bottom_seg = Line(self.grid['C4'] + UP*i*0.5 + LEFT*0.2, self.grid['E4'] + UP*i*0.5 + DOWN*0.5, color=WHITE).rotate(-45*DEGREES)
            bent_wavefronts.add(VGroup(top_seg, bottom_seg))

        self.play(FadeIn(wavefronts))
        self.play(wavefronts.animate.shift(DOWN * 1.5 + RIGHT * 0.5), run_time=2)
        
        # Transition to showing the bending effect
        self.play(ReplacementTransform(wavefronts, bent_wavefronts), run_time=1.5)
        self.wait(2)
        
        self.lecture[4].set_color(WHITE)
