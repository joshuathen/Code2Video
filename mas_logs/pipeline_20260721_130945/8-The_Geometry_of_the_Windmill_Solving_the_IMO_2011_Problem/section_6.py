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
        # Data
        lecture_lines = [
            "The invariant side-count ensures a repeating cycle.",
            "With finite stars, the windmill hits every point.",
            "Geometry transforms a simple rule into an infinite dance."
        ]
        
        # Setup
        self.setup_layout("Conclusion: The Infinite Loop", lecture_lines)
        
        # Assets - Lesson L009: Using designated assets
        laser_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg"
        star_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/star.svg"

        # Stars layout - Using Col 4 and 5 to avoid lecture notes (L003)
        # Pivot point set to C5 (Issue 37, 38)
        stars_pos = ["B4", "B5", "D4", "D5", "C4", "E4"]
        
        stars = VGroup()
        for pos in stars_pos:
            # SVGMobject for star as per storyboard
            star = SVGMobject(star_path).set_color(YELLOW).set_fill(YELLOW, opacity=0.8)
            self.place_at_grid(star, pos, scale_factor=0.2)
            stars.add(star)
            
        pivot_star = SVGMobject(star_path).set_color(WHITE).set_fill(WHITE, opacity=1)
        self.place_at_grid(pivot_star, "C5", scale_factor=0.15) # Scale factor as per Issue 37
        pivot_origin = self.grid["C5"]
        
        # Laser line as SVGMobject - Stretched to form a beam
        laser = SVGMobject(laser_path).set_color(RED)
        laser.stretch_to_fit_width(3.0)
        laser.stretch_to_fit_height(0.05)
        laser.move_to(pivot_origin)

        # === Animation for Lecture Line 1 ===
        # "The invariant side-count ensures a repeating cycle."
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # L011: Entry animations
        self.play(FadeIn(stars), FadeIn(pivot_star))
        self.play(Create(laser))
        
        # Use Rotate for performance and stability over updaters (L024)
        self.play(
            Rotate(laser, angle=2 * PI, about_point=pivot_origin),
            run_time=2,
            rate_func=rate_functions.ease_in_out_sine
        )
        
        # === Animation for Lecture Line 2 ===
        # "With finite stars, the windmill hits every point."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Storyboard: Hit each star, making it glow
        # L004: Using Indicate. L024: Simpler parallel animations to prevent timeout.
        self.play(
            Rotate(laser, angle=2 * PI, about_point=pivot_origin),
            LaggedStart(
                *[Indicate(s, color=WHITE, scale_factor=1.2) for s in stars],
                lag_ratio=0.15
            ),
            run_time=2.5,
            rate_func=linear
        )
        
        # === Animation for Lecture Line 3 ===
        # "Geometry transforms a simple rule into an infinite dance."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Storyboard: Fade out points and line to leave behind a history pattern
        # L024: Minimal count of history elements to ensure render budget.
        history = VGroup()
        for i in range(8):
            angle = i * (PI / 4)
            l = Line(LEFT * 1.5, RIGHT * 1.5, color=RED, stroke_width=0.5, stroke_opacity=0.3)
            l.rotate(angle)
            l.move_to(pivot_origin)
            history.add(l)
            
        self.play(
            FadeIn(history, lag_ratio=0.05),
            stars.animate.set_fill_opacity(0.1),
            pivot_star.animate.set_fill_opacity(0.1),
            laser.animate.set_stroke_opacity(0.1).set_fill_opacity(0.1),
            run_time=2
        )
        
        self.wait(1)
