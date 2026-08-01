from manim import *
import numpy as np
import os

class Section4Scene(Scene):
    def construct(self):
        # Configuration and Directory Setup
        os.makedirs(os.path.join("media", "texts"), exist_ok=True)
        self.camera.background_color = "#000000"

        # 1. Title and Layout Initialization
        title_text = "Bayes' Theorem: The Geometry of Logic"
        lecture_lines = [
            "- Prior Belief: P(H)",
            "- New Evidence: E",
            "- Likelihood: P(E|H)",
            "- Posterior: P(H|E)"
        ]
        
        self.display_sidebar(title_text, lecture_lines)
        self.visualize_bayes_geometry()
        self.wait(2)

    def display_sidebar(self, title_text, lecture_lines):
        # Title at the top
        title = Text(title_text, font_size=32, color=WHITE).to_edge(UP, buff=0.5)
        self.add(title)

        # Left-side sidebar content
        lecture_mobjects = [Text(line, font_size=24, color=GRAY_A) for line in lecture_lines]
        sidebar = VGroup(*lecture_mobjects).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        sidebar.to_edge(LEFT, buff=0.7)
        
        self.play(Create(sidebar), run_time=1.5)
        self.play(sidebar[0].animate.set_color(BLUE), sidebar[3].animate.set_color(YELLOW))

    def visualize_bayes_geometry(self):
        # Container for the visualization area (Right Side)
        viz_center = RIGHT * 3
        
        # Dimensions for probabilities
        sq_width = 5
        sq_height = 4
        prior_p_h = 0.4
        likelihood_e_h = 0.8
        likelihood_e_not_h = 0.2
        
        # Sample Space
        sample_space = Rectangle(width=sq_width, height=sq_height, color=WHITE).move_to(viz_center)
        
        # H and Not H areas
        h_rect = Rectangle(
            width=sq_width * prior_p_h, 
            height=sq_height, 
            fill_color=BLUE, 
            fill_opacity=0.3, 
            stroke_width=1
        ).align_to(sample_space, LEFT).align_to(sample_space, UP)
        
        not_h_rect = Rectangle(
            width=sq_width * (1 - prior_p_h), 
            height=sq_height, 
            fill_color=GRAY_C, 
            fill_opacity=0.1, 
            stroke_width=1
        ).align_to(sample_space, RIGHT).align_to(sample_space, UP)
        
        # Labels for H and Not H - Replaced MathTex with Text to avoid LaTeX dependency errors
        label_h = Text("H", color=BLUE, font_size=32).next_to(h_rect, UP)
        label_not_h = Text("¬H", color=GRAY_C, font_size=32).next_to(not_h_rect, UP)

        # Evidence areas (Intersection with H and Not H)
        e_intersect_h = Rectangle(
            width=sq_width * prior_p_h,
            height=sq_height * likelihood_e_h,
            fill_color=YELLOW,
            fill_opacity=0.7,
            stroke_width=0
        ).align_to(h_rect, BOTTOM).align_to(h_rect, LEFT)
        
        e_intersect_not_h = Rectangle(
            width=sq_width * (1 - prior_p_h),
            height=sq_height * likelihood_e_not_h,
            fill_color=YELLOW,
            fill_opacity=0.3,
            stroke_width=0
        ).align_to(not_h_rect, BOTTOM).align_to(not_h_rect, RIGHT)

        # Animation Sequence
        self.play(Create(sample_space))
        self.play(
            FadeIn(h_rect), FadeIn(not_h_rect),
            Write(label_h), Write(label_not_h)
        )
        self.wait(0.5)
        
        # Showing Evidence E
        e_label = Text("E", color=YELLOW, font_size=32).next_to(e_intersect_h, LEFT)
        self.play(
            FadeIn(e_intersect_h), 
            FadeIn(e_intersect_not_h),
            Write(e_label)
        )
        
        # Highlight Posterior P(H|E) - The part of E that belongs to H
        posterior_box = SurroundingRectangle(e_intersect_h, color=GOLD, buff=0.05)
        # Using Text for formula to ensure compatibility in environments without LaTeX
        posterior_math = Text("P(H|E) = P(E and H) / P(E)", font_size=26).next_to(sample_space, DOWN, buff=0.5)
        
        self.play(Create(posterior_box))
        self.play(Write(posterior_math))
        self.wait(1)