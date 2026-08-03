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

class Section1ThePizzaChallengeScene(TeachingScene):
    def construct(self):
        self.setup_layout("The Pizza Puzzle: Introduction", [
            "Meet Max, a baker cutting a circular pizza.",
            "He connects points on the crust with straight cuts.",
            "How many pieces can he make with these points?"
        ])

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        
        # Pizza SVG [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/pizza.svg]
        pizza = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pizza.svg", color=WHITE)
        pizza.set_height(3.0) 
        # Issue 25: Positioning pizza at B3-E6
        self.place_in_area(pizza, "B3", "E6")
        
        # Max the Baker icon (#FFD700)
        max_head = Circle(radius=0.15, color="#FFD700")
        max_body = Line(DOWN*0.15, DOWN*0.45, color="#FFD700")
        max_arms = Line(LEFT*0.25+DOWN*0.3, RIGHT*0.25+DOWN*0.3, color="#FFD700")
        max_legs = VGroup(
            Line(DOWN*0.45, LEFT*0.15+DOWN*0.75, color="#FFD700"),
            Line(DOWN*0.45, RIGHT*0.15+DOWN*0.75, color="#FFD700")
        )
        max_icon = VGroup(max_head, max_body, max_arms, max_legs)
        # Issue 26: place_at_grid(max_icon, 'B2', scale_factor=1.1)
        self.place_at_grid(max_icon, "B2", scale_factor=1.1)
        
        max_label = Text("Max", font_size=20, color="#FFD700")
        max_label.next_to(max_icon, DOWN, buff=0.1)
        
        self.play(DrawBorderThenFill(pizza), FadeIn(max_icon), Write(max_label), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(WHITE)
        
        # Use a reference circle for geometry logic
        pizza_ref = Circle(radius=1.5, stroke_opacity=0).move_to(pizza.get_center())
        
        # Points and Chord
        p1_angle = 45 * DEGREES
        p2_angle = 225 * DEGREES
        p1 = pizza_ref.point_at_angle(p1_angle)
        p2 = pizza_ref.point_at_angle(p2_angle)
        
        dot1 = Dot(p1, color=WHITE, radius=0.08)
        dot2 = Dot(p2, color=WHITE, radius=0.08)
        chord = Line(p1, p2, color=WHITE)
        
        self.play(Create(dot1), Create(dot2))
        self.play(Create(chord))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FF00")
        
        # Create regions for highlighting (#00FF00)
        # Segment 1
        region1 = Arc(radius=1.5, start_angle=p1_angle, angle=PI).shift(pizza_ref.get_center())
        region1.add_line_to(p1) 
        region1.set_fill("#00FF00", opacity=0.4)
        region1.set_stroke(width=0)
        
        # Segment 2
        region2 = Arc(radius=1.5, start_angle=p2_angle, angle=PI).shift(pizza_ref.get_center())
        region2.add_line_to(p2)
        region2.set_fill("#00FF00", opacity=0.4)
        region2.set_stroke(width=0)
        
        label1 = Text("1", font_size=36, color="#00FF00")
        label2 = Text("2", font_size=36, color="#00FF00")
        
        # Issue 27: label1 at C4, label2 at D5
        self.place_at_grid(label1, "C4")
        self.place_at_grid(label2, "D5")
        
        self.play(
            FadeIn(region1),
            FadeIn(region2),
            Write(label1),
            Write(label2)
        )
        self.wait(3)
