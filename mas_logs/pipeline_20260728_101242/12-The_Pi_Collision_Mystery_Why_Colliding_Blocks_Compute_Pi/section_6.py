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
        # Initial Setup
        title_str = "Summary: The Geometry of Physics"
        lines = [
            "Elastic collisions are hidden geometric operations.",
            "Pi emerges from the fundamental symmetry of energy.",
            "A simple physical system computes nature's most famous constant."
        ]
        self.setup_layout(title_str, lines)

        # === Animation for Lecture Line 1 ===
        # "Elastic collisions are hidden geometric operations."
        self.lecture[0].set_color(YELLOW)

        # Blocks Region (Issue 33: Row B)
        ground = Line(self.grid["B1"] + LEFT*0.5, self.grid["B6"] + RIGHT*0.5, color=GREY)
        wall = Line(self.grid["A1"] + LEFT*0.5, self.grid["B1"] + LEFT*0.5, color=GREY)
        
        block1 = Square(side_length=0.6, fill_opacity=0.8, fill_color="#00FFFF", color="#00FFFF")
        block2 = Square(side_length=1.0, fill_opacity=0.8, fill_color="#FF00FF", color="#FF00FF")
        
        # Position blocks on Row B (Issue 33)
        self.place_at_grid(block1, "B2")
        self.place_at_grid(block2, "B5")
        block1.shift(UP * 0.3)
        block2.shift(UP * 0.5)
        
        blocks_group = VGroup(ground, wall, block1, block2)

        # Circle Region (Issue 34: C1 to F6)
        circle = Circle(radius=1.0, color=WHITE)
        self.place_in_area(circle, "C1", "F6", scale_factor=1.2)
        
        # Persistent State Dot with ValueTracker
        angle_tracker = ValueTracker(0)
        dot = Dot(color="#FFFF00")
        dot.add_updater(lambda d: d.move_to(circle.point_at_angle(angle_tracker.get_value())))
        
        visuals_group = VGroup(blocks_group, circle)
        self.play(FadeIn(visuals_group), FadeIn(dot))
        
        # Show colliding motion and reflecting dot
        self.play(
            block2.animate.shift(LEFT * 1.5),
            block1.animate.shift(LEFT * 0.4),
            angle_tracker.animate.set_value(PI / 2),
            run_time=1.5,
            rate_func=linear
        )
        self.play(
            block2.animate.shift(RIGHT * 0.5),
            block1.animate.shift(RIGHT * 0.1),
            angle_tracker.animate.set_value(PI),
            run_time=1.5,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Pi emerges from the fundamental symmetry of energy."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Load Pi symbol
        pi_sym = MathTex(r"\pi", color=WHITE, font_size=72)
        pi_sym.move_to(circle.get_center())

        # Make circle glow and show Pi
        self.play(
            circle.animate.set_color("#FFFF00").set_stroke(width=8),
            Write(pi_sym),
            run_time=1.0
        )
        self.play(Flash(circle, color="#FFFF00", line_length=0.3, flash_radius=1.3, num_lines=12))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # "A simple physical system computes nature's most famous constant."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Fade out visuals to leave only a large Pi symbol (Issue 32)
        self.play(
            FadeOut(blocks_group),
            FadeOut(circle),
            FadeOut(dot),
            run_time=1
        )
        
        # Shift and scale the Pi symbol as requested in Issue 32
        # Action: Shift the large Pi symbol further right and down by using self.place_in_area(pi_sym, 'B2', 'F6', scale_factor=0.8)
        # We'll calculate the target position/scale and animate to it.
        target_pi = MathTex(r"\pi", color=WHITE, font_size=200) # Start large for the "large" requirement
        self.place_in_area(target_pi, "B2", "F6", scale_factor=0.8)
        
        self.play(
            pi_sym.animate.move_to(target_pi.get_center()).scale_to_fit_height(target_pi.height),
            run_time=1.5
        )
        self.wait(2)
