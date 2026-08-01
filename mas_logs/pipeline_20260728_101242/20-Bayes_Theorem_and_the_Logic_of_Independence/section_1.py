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
        self.setup_layout("The Hook: The Detective's Dilemma", [
            "Probability helps us update beliefs with new evidence.",
            "It's a dynamic tool for logical reasoning.",
            "Does knowing it's rainy change the butterfly's odds?"
        ])
        
        # Initial state: Dim lecture lines
        for line in self.lecture:
            line.set_color("#666666")

        # === Animation for Lecture Line 1 ===
        # Matching color: WHITE
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Construct magnifying glass
        glass_circle = Circle(radius=0.4, color=WHITE, stroke_width=4)
        glass_handle = Rectangle(height=0.1, width=0.5, color=WHITE, fill_opacity=1).rotate(-PI/4)
        magnifying_glass = VGroup(glass_circle, glass_handle)
        glass_handle.move_to(glass_circle.get_center() + DR * 0.4)
        
        # Issue 31: Relocate to B4 to be closer to other icons
        self.place_at_grid(magnifying_glass, 'B4', scale_factor=1.0)
        self.play(FadeIn(magnifying_glass))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Matching color: WHITE
        self.play(
            self.lecture[0].animate.set_color("#666666"),
            self.lecture[1].animate.set_color(WHITE)
        )
        
        # Probability Slider
        slider_line = Line(LEFT, RIGHT, color=WHITE).scale(1.2)
        low_label = Text("Low", font_size=18, color=WHITE).next_to(slider_line, LEFT)
        high_label = Text("High", font_size=18, color=WHITE).next_to(slider_line, RIGHT)
        knob = Dot(color=WHITE, radius=0.12)
        slider = VGroup(slider_line, low_label, high_label, knob)
        
        # Issue 30: Use place_in_area for better balance
        self.place_in_area(slider, 'E3', 'E5', scale_factor=1.0)
        # Position knob relative to the line after the group has moved
        knob.move_to(slider_line.get_start())
        
        self.play(FadeIn(slider))
        self.play(knob.animate.move_to(slider_line.get_end()), run_time=1.2, rate_func=there_and_back)
        self.play(knob.animate.move_to(slider_line.get_center()), run_time=0.8)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Matching color: CYAN (#00FFFF)
        self.play(
            self.lecture[1].animate.set_color("#666666"),
            self.lecture[2].animate.set_color("#00FFFF")
        )
        
        # Butterfly
        wing_l = Ellipse(width=0.4, height=0.7, color="#00FFFF", fill_opacity=0.7).rotate(PI/6)
        wing_r = Ellipse(width=0.4, height=0.7, color="#00FFFF", fill_opacity=0.7).rotate(-PI/6)
        body = Line(UP*0.25, DOWN*0.25, color=WHITE, stroke_width=3)
        butterfly = VGroup(wing_l, wing_r, body)
        wing_l.next_to(body, LEFT, buff=-0.1)
        wing_r.next_to(body, RIGHT, buff=-0.1)
        
        # Rain cloud
        c1 = Circle(radius=0.25, color="#888888", fill_opacity=1)
        c2 = Circle(radius=0.35, color="#888888", fill_opacity=1).shift(RIGHT*0.25)
        c3 = Circle(radius=0.25, color="#888888", fill_opacity=1).shift(RIGHT*0.5)
        cloud = VGroup(c1, c2, c3).move_to(ORIGIN)
        
        # Move objects to their positions
        # Issue 32: Consistent scale factor
        self.place_at_grid(butterfly, 'C5', scale_factor=1.0)
        self.place_at_grid(cloud, 'B5', scale_factor=1.0)
        
        self.play(FadeIn(butterfly))
        self.wait(0.5)
        self.play(FadeIn(cloud))
        # Change butterfly color to dark blue
        self.play(butterfly.animate.set_color("#00008B"))
        self.wait(2)
