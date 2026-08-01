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
        # Setup layout with title and lecture lines
        # Replaced LaTeX delimiters for plain Text rendering to avoid 'latex' executable requirement
        title_text = "The Grand Finale: e^(iπ) = -1"
        lecture_lines = [
            'Let the growth continue for a distance of π.',
            'We travel exactly halfway around the circle.',
            'We end up on the opposite side of one.',
            'This path of length π leads to negative one.',
            'We have arrived at the identity e^(iπ) = -1.'
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Pre-define visual elements ===
        
        # Replaced MathTex with Text to resolve FileNotFoundError: 'latex'
        formula_sub = Text("e^(iπ) = cos(π) + i sin(π)", color="#FFFFFF", font_size=32)
        self.place_in_area(formula_sub, "A1", "A6", scale_factor=0.65)

        # Complex Plane visual
        plane_center_coord = self.grid["D4"]
        axes = ComplexPlane(
            x_range=[-1.5, 1.5, 1],
            y_range=[-1.5, 1.5, 1],
            x_length=3,
            y_length=3,
            axis_config={"include_tip": True, "color": GREY_B}
        ).move_to(plane_center_coord)
        
        unit_circle = Circle(radius=1.5, color=GREY_D).move_to(plane_center_coord)
        
        # Vector and Arc logic using ValueTracker
        vec_tracker = ValueTracker(0)
        
        vector = Arrow(
            start=plane_center_coord,
            end=plane_center_coord + RIGHT * 1.5,
            buff=0,
            color=WHITE,
            stroke_width=4
        )
        vector.add_updater(lambda m: m.become(Arrow(
            start=plane_center_coord,
            end=plane_center_coord + np.array([np.cos(vec_tracker.get_value()), np.sin(vec_tracker.get_value()), 0]) * 1.5,
            buff=0,
            color=WHITE,
            stroke_width=4
        )))
        
        arc = Arc(
            radius=1.5,
            start_angle=0,
            angle=0.001,
            color="#FFFF00",
            stroke_width=5
        ).shift(plane_center_coord)
        arc.add_updater(lambda m: m.become(Arc(
            radius=1.5,
            start_angle=0,
            angle=max(0.001, vec_tracker.get_value()),
            color="#FFFF00",
            stroke_width=5
        ).shift(plane_center_coord)))

        # Points and Labels
        start_dot = Dot(plane_center_coord + RIGHT * 1.5, color=WHITE)
        start_label = Text("1", font_size=20, color=WHITE)
        self.place_at_grid(start_label, "D6", scale_factor=0.8) 
        
        end_dot = Dot(plane_center_coord + LEFT * 1.5, color="#FF0000")
        end_label = Text("-1", font_size=20, color="#FF0000")
        self.place_at_grid(end_label, "D2", scale_factor=0.8) 

        # arc_label: Replaced MathTex with Text
        arc_label = Text("π", color="#FFFF00", font_size=32)
        self.place_at_grid(arc_label, "B4", scale_factor=0.9)

        # conclusion: Replaced MathTex with Text
        conclusion = Text("e^(iπ) = -1", color=WHITE, font_size=42)
        self.place_in_area(conclusion, "F1", "F6", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(Write(formula_sub))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(WHITE))
        self.play(Create(axes), Create(unit_circle))
        self.play(FadeIn(start_dot), Write(start_label))
        self.add(vector, arc)
        self.play(vec_tracker.animate.set_value(PI), run_time=3, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF0000"))
        self.play(FadeIn(end_dot), Write(end_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFFF00"))
        self.play(Write(arc_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.play(Write(conclusion))
        self.wait(3)
