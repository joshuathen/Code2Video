from manim import *
import math

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # Set background color
        self.camera.background_color = "#000000"
        
        # Title setup
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content
        # Using VGroup and arranging to the left
        lecture_texts = [Text(f"• {line}", font_size=20, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        self.lecture.to_edge(LEFT, buff=0.5).shift(DOWN * 0.5)
        self.add(self.lecture)
        
        # Create a workspace area on the right half of the screen
        self.workspace = Rectangle(
            width=6, height=6, stroke_width=0
        ).to_edge(RIGHT, buff=0.5)

class Section3Scene(TeachingScene):
    def construct(self):
        # Define the lecture content
        lines = [
            "Unit sphere volume behaves strangely as n increases.",
            "It peaks at five dimensions and then drops.",
            "For very high dimensions, the volume approaches zero.",
            "Compare this to a unit cube's constant volume.",
            "High-dimensional spheres are surprisingly 'empty'."
        ]
        
        self.setup_layout("The Volume Paradox", lines)

        pass