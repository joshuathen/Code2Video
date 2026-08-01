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
        
        # Assets - Lesson L009: Use designated SVG paths
        laser_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg"
        star_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/star.svg"

        # Lesson L003: Positioning. Moving pivot to C5 and stars to Col 4-5 to avoid lecture text (Issue 37, 38).
        # We avoid Column 6 and Row F to prevent clipping and stay clear of Row A reserved for titles.
        stars_pos = ["B4", "B5", "C4", "D4", "D5", "E4", "E5"]
        
        stars = VGroup()
        for pos in stars_pos:
            star = SVGMobject(star_path).set_color(YELLOW).set_fill(YELLOW, opacity=0.8)
            self.place_at_grid(star, pos, scale_factor=0.25)
            stars.add(star)
            
        pivot_star = SVGMobject(star_path).set_color(WHITE).set_fill(WHITE, opacity=1)
        # Fix for Issue 37/38: Use suggested pivot at C5 to utilize right-side space
        self.place_at_grid(pivot_star, "C5", scale_factor=0.3)
        pivot_origin = self.grid["C5"]
        
        # Laser line as SVGMobject
        laser = SVGMobject(laser_path).set_color(RED)
        # Stretch it to look like a long beam. 
        laser.stretch_to_fit_width(4)
        laser.stretch_to_fit_height(0.08)
        laser.move_to(pivot_origin)
        laser.save_state()
        
        # Tracking variables
        angle_tracker = ValueTracker(0)
        
        # Lesson L008: Use updaters and ValueTracker for stable movement
        def update_laser(m):
            angle = angle_tracker.get_value()
            m.restore()
            m.rotate(angle, about_point=pivot_origin)

        # === Animation for Lecture Line 1 ===
        # "The invariant side-count ensures a repeating cycle."
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Lesson L011: Introduce mobjects via entry animations
        self.play(FadeIn(stars), FadeIn(pivot_star))
        self.play(Create(laser))
        
        laser.add_updater(update_laser)
        
        # Accelerate rotation to demonstrate cycling
        self.play(
            angle_tracker.animate.set_value(4 * PI),
            run_time=4,
            rate_func=rate_functions.ease_in_sine
        )
        
        # === Animation for Lecture Line 2 ===
        # "With finite stars, the windmill hits every point."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Stars glow as laser "hits" them
        # Lesson L004: Use Indicate
        indication_anims = []
        star_indices = [2, 4, 0, 6, 1, 5, 3] # Fixed sequence for deterministic visual flow
        for i, idx in enumerate(star_indices):
            delay = i * (3.5 / len(stars))
            indication_anims.append(
                Succession(
                    Wait(delay),
                    Indicate(stars[idx], color=WHITE, scale_factor=1.4)
                )
            )

        self.play(
            angle_tracker.animate.set_value(8 * PI),
            AnimationGroup(*indication_anims),
            run_time=4,
            rate_func=linear
        )
        
        # === Animation for Lecture Line 3 ===
        # "Geometry transforms a simple rule into an infinite dance."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Create a history trail of the line to leave a "flower-like" pattern
        # Lesson L024: Simplify complexity for render stability
        history = VGroup()
        for i in range(24): 
            angle = i * (2 * PI / 24)
            # Create a symmetric pattern with a sinusoidal offset
            offset_mag = 0.5 * np.cos(3 * angle) 
            offset = offset_mag * np.array([np.cos(angle), np.sin(angle), 0])
            
            # Using Line for history trail to keep geometry lightweight
            l = Line(LEFT * 2, RIGHT * 2, color=RED, stroke_width=0.6, stroke_opacity=0.2)
            l.rotate(angle)
            l.move_to(pivot_origin + offset)
            history.add(l)
            
        self.play(
            FadeIn(history, lag_ratio=0.1),
            stars.animate.set_fill_opacity(0.15),
            pivot_star.animate.set_fill_opacity(0.15),
            laser.animate.set_stroke_opacity(0.1).set_fill_opacity(0.1),
            run_time=5
        )
        
        self.wait(3)
