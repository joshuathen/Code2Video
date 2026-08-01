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
        self.setup_layout(
            "The Detective's Dilemma: Introducing Prior Belief",
            [
                "Bayes' Theorem helps us update beliefs using new evidence.",
                "We begin with a Prior Belief before seeing data.",
                "Imagine a Ranger seeking a rare Blue Phoenix."
            ]
        )
        
        # Colors
        COLOR_FEATHER = "#00FFFF"
        COLOR_RANGER = "#FFFFFF"
        COLOR_PRIOR = "#ADD8E6"
        COLOR_HIGHLIGHT = YELLOW

        # === Animation for Lecture Line 1 ===
        # Display a blue feather icon (#00FFFF) [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/feather.svg]
        # and a ranger silhouette (#FFFFFF) on the right grid.
        
        # Load asset for feather (Issue 19)
        feather = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/feather.svg").set_color(COLOR_FEATHER)
        feather_label = Text("Feather", font_size=16, color=COLOR_FEATHER)
        # Position feather at B5 to utilize right side (Issue 22)
        self.place_at_grid(feather, "B5", scale_factor=0.5) 
        feather_label.next_to(feather, DOWN, buff=0.1)
        
        ranger = Square(side_length=0.6, color=COLOR_RANGER, fill_opacity=0.5)
        # Add a "head" to the square to make it look slightly more like a silhouette
        ranger_head = Circle(radius=0.15, color=COLOR_RANGER, fill_opacity=0.8)
        ranger_head.next_to(ranger, UP, buff=0)
        ranger_sil = VGroup(ranger, ranger_head)
        ranger_label = Text("Ranger", font_size=16, color=COLOR_RANGER)
        # Position ranger at B3 to avoid crowding lecture notes (Issue 21)
        self.place_at_grid(ranger_sil, "B3", scale_factor=0.8)
        ranger_label.next_to(ranger_sil, DOWN, buff=0.1)
        
        self.play(
            FadeIn(feather), FadeIn(feather_label),
            FadeIn(ranger_sil), FadeIn(ranger_label),
            self.lecture[0].animate.set_color(COLOR_HIGHLIGHT)
        )
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Show 'Prior P(H) = 1%' text in light blue (#ADD8E6) near the ranger.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_PRIOR)
        )
        
        prior_eq = MathTex("P(H) = ", color=COLOR_PRIOR, font_size=36)
        prior_val = MathTex("1\\%", color=COLOR_PRIOR, font_size=36)
        prior_group = VGroup(prior_eq, prior_val).arrange(RIGHT, buff=0.1)
        # Positioning prior_group in area D3-D5 for better balance (Issue 23)
        self.place_in_area(prior_group, 'D3', 'D5', scale_factor=0.9)
        
        prior_label = Text("Prior Belief", font_size=20, color=COLOR_PRIOR)
        prior_label.next_to(prior_group, UP, buff=0.2)
        
        self.play(
            Write(prior_group),
            Write(prior_label)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Scale the '1%' text up and down to emphasize our starting guess.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_HIGHLIGHT)
        )
        
        self.play(
            prior_val.animate.scale(1.5),
            run_time=0.6,
            rate_func=there_and_back
        )
        self.wait(3)
