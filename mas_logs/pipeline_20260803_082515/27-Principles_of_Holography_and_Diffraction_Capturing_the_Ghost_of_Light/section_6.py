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

class Section6Scene(TeachingScene):
    def construct(self):
        title = "Real-World Diffraction: From Credit Cards to VR"
        lines = [
            "Tiny diffraction gratings provide security on credit cards.",
            "Head-up displays use these principles to project transparent data.",
            "Holography brings 3D visuals to modern engineering and art."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Initial state: highlight first line in red to match bird
        self.play(self.lecture[0].animate.set_color("#FF0000"))
        
        # Create a silver credit card
        card = RoundedRectangle(
            corner_radius=0.1, height=1.4, width=2.2, 
            fill_opacity=1, fill_color="#C0C0C0", stroke_color=WHITE
        )
        
        # Bird shape for hologram
        bird_body = Circle(radius=0.08, fill_opacity=1, color="#FF0000")
        wing_l = Arc(radius=0.2, start_angle=0.5*PI, angle=PI, color="#FF0000").rotate(0.5*PI)
        wing_r = Arc(radius=0.2, start_angle=0.5*PI, angle=PI, color="#FF0000").rotate(-0.5*PI)
        wing_l.shift(LEFT*0.08)
        wing_r.shift(RIGHT*0.08)
        bird = VGroup(bird_body, wing_l, wing_r).scale(0.5)
        # Positioning bird relative to card
        bird.shift(RIGHT*0.6 + UP*0.35)
        
        card_group = VGroup(card, bird)
        # Fix Issue 48: Use A4-B6 to avoid overlap with lower elements
        self.place_in_area(card_group, 'A4', 'B6')
        
        self.play(FadeIn(card_group))
        self.wait(0.5)
        
        # Card tilts and bird color shifts smoothly from red to blue
        self.play(
            Rotate(card_group, angle=0.4, axis=RIGHT),
            bird.animate.set_color("#0000FF"),
            self.lecture[0].animate.set_color("#0000FF"),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#98FB98")
        )
        
        # VR lens cross-section (stretched circle)
        lens_body = Circle(radius=0.8, color="#ADD8E6", fill_opacity=0.3).stretch(0.3, dim=0)
        
        # Tiny gratings shown on lens surface
        gratings = VGroup(*[
            Line(UP*0.15, DOWN*0.15, color="#98FB98", stroke_width=2).shift(RIGHT*i*0.08) 
            for i in range(-3, 4)
        ]).move_to(lens_body.get_center())
        
        # Light rays bending through the lens
        ray_in = Line(LEFT*1.2, LEFT*0.2, color=WHITE).add_tip(tip_length=0.1)
        ray_out_up = Line(LEFT*0.2, RIGHT*0.8 + UP*0.4, color=WHITE).add_tip(tip_length=0.1)
        ray_out_down = Line(LEFT*0.2, RIGHT*0.8 + DOWN*0.4, color=WHITE).add_tip(tip_length=0.1)
        rays = VGroup(ray_in, ray_out_up, ray_out_down).move_to(lens_body.get_center())
        
        hud_group = VGroup(lens_body, gratings, rays)
        # Fix Issue 49: Use C4-D6 to keep it separated from card and holo
        self.place_in_area(hud_group, 'C4', 'D6', scale_factor=0.8)
        
        self.play(FadeIn(hud_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(GOLD)
        )
        
        # Holography visual: A rotating 3D star-like object
        holo_shape = Star(n=6, outer_radius=0.8, inner_radius=0.4, color=GOLD, fill_opacity=0.7)
        # Fix Issue 50: Use E4-F6 for bottom positioning
        self.place_in_area(holo_shape, 'E4', 'F6')
        
        self.play(FadeIn(holo_shape))
        self.play(Rotate(holo_shape, angle=TAU, run_time=2))
        self.wait(2)
