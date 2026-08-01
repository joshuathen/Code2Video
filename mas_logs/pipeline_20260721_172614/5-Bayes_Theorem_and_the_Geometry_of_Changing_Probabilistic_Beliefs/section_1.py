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
        # Data from storyboard
        title_text = "The Hook: The Detective's Dilemma"
        lecture_lines = [
            "Meet Sherlock Bones, a detective seeking hidden treasure.",
            "He starts with an initial belief called a \"Prior\".",
            "Visualizing probability as space makes updating beliefs intuitive."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        COLOR_GOLD = "#FFD700"
        COLOR_BLUE = "#87CEEB"
        
        # === Animation for Lecture Line 1 ===
        # Line 1: Highlighted in a distinct light color (Sky Blue)
        self.play(self.lecture[0].animate.set_color(COLOR_BLUE))
        
        # Use Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/detect.svg
        try:
            detective = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/detect.svg")
            detective.set_color(COLOR_BLUE)
        except:
            # Fallback representation of Sherlock Bones
            detective = VGroup(
                Circle(radius=0.5, color=COLOR_BLUE, fill_opacity=0.8),
                Rectangle(height=0.2, width=0.8, color="#5D8AA8", fill_opacity=1).shift(UP*0.4),
                Text("Bones", font_size=16).shift(DOWN*0.7)
            )
            
        # Fix: Issue 28 - Place at A2
        self.place_at_grid(detective, 'A2', scale_factor=0.8)
        self.play(DrawBorderThenFill(detective))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Line 2: Highlighted in Gold to match the Prior Bar
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_GOLD)
        )
        
        # Prior Belief Bar: 20% in Gold (#FFD700)
        bar_bg = Rectangle(height=0.4, width=3.0, color=GREY, stroke_width=2)
        bar_fill = Rectangle(height=0.4, width=3.0 * 0.2, color=COLOR_GOLD, fill_opacity=1, stroke_width=0).align_to(bar_bg, LEFT)
        bar_label = Text("Prior Belief: 20%", font_size=20, color=COLOR_GOLD)
        
        prior_group = VGroup(bar_bg, bar_fill, bar_label).arrange(DOWN, buff=0.3)
        # Fix: Issue 29 - Place at A5
        self.place_at_grid(prior_group, 'A5', scale_factor=0.8)
        
        self.play(Create(bar_bg), FadeIn(bar_fill), Write(bar_label))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Line 3: Highlighted in White to match the Universe Square
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(WHITE)
        )
        
        # Large Square representing 'Universe of Possibilities'
        universe_square = Square(side_length=3.5, color=WHITE, stroke_width=2)
        universe_label = Text("Universe of Possibilities", font_size=24, color=WHITE)
        universe_group = VGroup(universe_square, universe_label).arrange(UP, buff=0.4)
        
        # Fix: Issue 30 - Place in C2 to F5
        self.place_in_area(universe_group, 'C2', 'F5', scale_factor=0.9)
        
        # Transition: Shift the view to the Universe of Possibilities
        self.play(
            FadeOut(detective),
            FadeOut(prior_group),
            Create(universe_square),
            Write(universe_label)
        )
        self.wait(3)
        
        # Reset colors for final wait
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)
