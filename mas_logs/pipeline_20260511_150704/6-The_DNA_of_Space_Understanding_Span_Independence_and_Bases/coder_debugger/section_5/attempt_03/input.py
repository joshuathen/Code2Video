from manim import *
import numpy as np

class Section5Scene(Scene):
    def construct(self):
        # Configuration
        self.camera.background_color = "#000000"
        
        # 1. Title and Layout Setup
        title = Text("The DNA of Space: Span & Bases", font_size=32, color=WHITE).to_edge(UP, buff=0.5)
        
        # Left side: Lecture Points
        lecture_content = VGroup(
            Text("- Linear Span", font_size=22),
            Text("- Linear Independence", font_size=22),
            Text("- Identifying a Basis", font_size=22),
            Text("- Dimension of Space", font_size=22)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.6)
        lecture_content.to_edge(LEFT, buff=1.0).shift(DOWN * 0.2)

        # 2. Right side: Coordinate System Visualization
        # We create a coordinate system relative to a specific center point on the right
        plane_center = RIGHT * 3 + DOWN * 0.5
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=5,
            y_length=5,
            background_line_style={
                "stroke_color": BLUE_E,
                "stroke_width": 1,
                "stroke_opacity": 0.5
            }
        ).move_to(plane_center)

        # Basis Vectors
        vec_i = Arrow(
            start=plane.coords_to_point(0, 0),
            end=plane.coords_to_point(1, 0),
            buff=0,
            color=YELLOW,
            stroke_width=4
        )
        vec_j = Arrow(
            start=plane.coords_to_point(0, 0),
            end=plane.coords_to_point(0, 1),
            buff=0,
            color=PINK,
            stroke_width=4
        )

        # Using Text instead of MathTex to avoid the LaTeX dependency error
        label_i = Text("v1", font_size=24, color=YELLOW).next_to(vec_i.get_end(), DOWN, buff=0.1)
        label_j = Text("v2", font_size=24, color=PINK).next_to(vec_j.get_end(), LEFT, buff=0.1)

        # 3. Animations
        self.play(Write(title))
        self.wait(0.5)

        # Animate bullet points one by one
        for point in lecture_content:
            self.play(FadeIn(point, shift=RIGHT), run_time=0.7)

        # Create Coordinate System
        self.play(Create(plane), run_time=1.5)
        self.play(
            GrowArrow(vec_i),
            GrowArrow(vec_j),
            Write(label_i),
            Write(label_j)
        )
        self.wait(0.5)

        # Visualizing Span (Highlighting the plane area)
        span_area = Rectangle(
            width=plane.x_length, 
            height=plane.y_length, 
            fill_color=BLUE, 
            fill_opacity=0.15, 
            stroke_width=0
        ).move_to(plane_center)
        
        span_label = Text("Span{v1, v2}", font_size=20, color=BLUE_B).next_to(plane, DOWN, buff=0.3)

        self.play(
            FadeIn(span_area),
            Write(span_label)
        )
        
        # Highlight specific grid coordinate logic
        target_dot = Dot(plane.coords_to_point(2, 1), color=WHITE, radius=0.06)
        # Replacing MathTex with Text to resolve FileNotFoundError: 'latex'
        target_label = Text("2v1 + 1v2", font_size=18).next_to(target_dot, UR, buff=0.1)
        
        self.play(
            Create(target_dot),
            Write(target_label)
        )

        self.wait(3)