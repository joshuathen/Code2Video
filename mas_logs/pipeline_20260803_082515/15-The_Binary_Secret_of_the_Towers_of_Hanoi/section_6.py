from manim import *
import numpy as np
import random

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
        self.setup_layout(
            "Conclusion: Mathematical Elegance", 
            [
                "Abstract binary numbers perfectly map to this physical puzzle.",
                "Solution takes exactly two to the n minus one moves.",
                "Mathematics reveals hidden order in legendary complexity."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Color: #00FFFF (Cyan)
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        
        # Show a digital counter iterating rapidly from 1 to 2^n-1 in #00FFFF (Cyan)
        # Using ValueTracker for efficient numerical updates
        counter_val = ValueTracker(1)
        counter_num = DecimalNumber(
            1, 
            num_decimal_places=0, 
            group_with_commas=True, 
            font_size=40,
            color="#00FFFF"
        )
        # Fix Issue 34: self.place_in_area(counter_num, 'C3', 'D6', scale_factor=1.0)
        self.place_in_area(counter_num, "C3", "D6", scale_factor=1.0)
        
        # updater to change value in place
        counter_num.add_updater(lambda m: m.set_value(counter_val.get_value()))
        
        self.add(counter_num)
        # Iterate to a large number representing 2^n-1 scalability
        self.play(counter_val.animate.set_value(10**14), run_time=3, rate_func=linear)
        self.wait(1)
        self.play(FadeOut(counter_num))

        # === Animation for Lecture Line 2 ===
        # Color: #00FF00 (Green)
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        # Display the general formula 'Moves = 2^n - 1' in large, centered #00FF00 (Green) text.
        formula = MathTex("Moves = 2^n - 1", color="#00FF00", font_size=60)
        # Fix Issue 35: self.place_in_area(formula, 'C3', 'E6', scale_factor=1.0)
        self.place_in_area(formula, "C3", "E6", scale_factor=1.0)
        
        self.play(Write(formula))
        self.wait(2)
        self.play(FadeOut(formula))

        # === Animation for Lecture Line 3 ===
        # Color: #FFFFFF (White)
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        
        # Fade out all elements into a starry background while a silhouette of Professor Hoot appears in the center.
        
        # Create stars
        stars = VGroup(*[
            Dot(
                point=[random.uniform(0.1, 6.5), random.uniform(-3.5, 3.5), 0],
                radius=random.uniform(0.01, 0.04),
                color=WHITE,
                fill_opacity=random.uniform(0.2, 0.7)
            ) for _ in range(100)
        ])
        
        # Fix Issue 27: Use Asset for Professor Hoot
        hoot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/owl.svg")
        hoot.set_color("#FFFFFF") # Silhouette color or white for visibility
        hoot.set_fill(opacity=1)
        
        # Fix Issue 36: self.place_in_area(hoot, 'C4', 'D5', scale_factor=1.2)
        self.place_in_area(hoot, "C4", "D5", scale_factor=1.2)

        self.play(
            FadeOut(self.title),
            FadeOut(self.lecture),
            FadeIn(stars),
            FadeIn(hoot),
            run_time=2
        )
        self.wait(3)
