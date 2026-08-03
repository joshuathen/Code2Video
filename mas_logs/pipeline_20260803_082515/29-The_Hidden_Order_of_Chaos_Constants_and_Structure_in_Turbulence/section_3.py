from manim import *

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

class Section3Scene(TeachingScene):
    def construct(self):
        title = "The Energy Cascade: Richardson’s Vision"
        lines = [
            "Energy enters the system at the largest scales.",
            "Large eddies break into smaller and smaller swirls.",
            "Richardson described this as a continuous energy cascade.",
            "Kinetic energy transfers down without being lost.",
            "This process continues until the smallest scales."
        ]
        self.setup_layout(title, lines)

        # Colors
        YELLOW_C = "#FFFF00"
        SILVER_C = "#C0C0C0"
        WHITE_C = "#FFFFFF"

        # Rotation tracker to control global eddy speed
        rot_tracker = ValueTracker(0)

        # === Animation for Lecture Line 1 ===
        # "Energy enters the system at the largest scales."
        self.lecture[0].set_color(YELLOW_C)
        
        large_circle = Circle(radius=1.8, color=YELLOW_C, stroke_width=4)
        large_arrow = Arrow(start=UP*1.2, end=DOWN*1.2, color=YELLOW_C, buff=0).rotate(PI/4)
        large_eddy = VGroup(large_circle, large_arrow)
        
        # Resolved Issue 31: Positioned further right to avoid lecture text overlap
        self.place_in_area(large_eddy, "B3", "E6", scale_factor=0.9)
        
        # Persistent rotation via updater
        large_eddy.add_updater(lambda m, dt: m.rotate(-rot_tracker.get_value() * dt))
        
        self.play(Create(large_eddy), run_time=2)
        self.play(rot_tracker.animate.set_value(2), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Large eddies break into smaller and smaller swirls."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(SILVER_C)
        
        # Create smaller silver circles representing eddies breaking down
        small_circle1 = Circle(radius=0.7, color=SILVER_C, stroke_width=3)
        small_arrow1 = Arrow(start=UP*0.5, end=DOWN*0.5, color=SILVER_C, buff=0).rotate(PI/4)
        small_eddy1 = VGroup(small_circle1, small_arrow1)
        
        small_circle2 = Circle(radius=0.7, color=SILVER_C, stroke_width=3)
        small_arrow2 = Arrow(start=UP*0.5, end=DOWN*0.5, color=SILVER_C, buff=0).rotate(PI/4)
        small_eddy2 = VGroup(small_circle2, small_arrow2)
        
        # Resolved Issue 32 & 33: Positioned at B3 and E6 for better distribution
        self.place_at_grid(small_eddy1, "B3", scale_factor=0.8)
        self.place_at_grid(small_eddy2, "E6", scale_factor=0.8)
        
        # Smaller eddies rotate faster
        small_eddy1.add_updater(lambda m, dt: m.rotate(-rot_tracker.get_value() * 1.5 * dt))
        small_eddy2.add_updater(lambda m, dt: m.rotate(-rot_tracker.get_value() * 1.5 * dt))
        
        self.play(
            FadeIn(small_eddy1, scale=0.5),
            FadeIn(small_eddy2, scale=0.5),
            large_eddy.animate.set_stroke(opacity=0.3).scale(0.9),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Richardson described this as a continuous energy cascade."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(WHITE_C)
        
        # Visualizing the 'cascade' with flow lines from center to outer eddies
        cascade_arrow1 = CurvedArrow(self.grid["D4"], self.grid["B3"], color=WHITE_C, angle=-PI/4)
        cascade_arrow2 = CurvedArrow(self.grid["D5"], self.grid["E6"], color=WHITE_C, angle=-PI/4)
        
        self.play(Create(cascade_arrow1), Create(cascade_arrow2), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Kinetic energy transfers down without being lost."
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW_C)
        
        # Pulsing effect to emphasize energy conservation/transfer
        self.play(
            large_eddy.animate.set_stroke(width=6, opacity=0.8),
            small_eddy1.animate.set_stroke(width=6),
            small_eddy2.animate.set_stroke(width=6),
            run_time=0.8
        )
        self.play(
            large_eddy.animate.set_stroke(width=4, opacity=0.3),
            small_eddy1.animate.set_stroke(width=3),
            small_eddy2.animate.set_stroke(width=3),
            run_time=0.8
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "This process continues until the smallest scales."
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(WHITE_C)
        
        # Tiny white eddies appearing around the silver ones (B3 and E6)
        tiny_eddies = VGroup()
        tiny_positions = ["A3", "A4", "B4", "C3", "D5", "D6", "E5", "F6"]
        for pos in tiny_positions:
            t_circle = Circle(radius=0.15, color=WHITE_C, stroke_width=1.5)
            self.place_at_grid(t_circle, pos)
            tiny_eddies.add(t_circle)
            
        self.play(
            FadeIn(tiny_eddies, shift=DOWN),
            FadeOut(cascade_arrow1),
            FadeOut(cascade_arrow2),
            run_time=2
        )
        
        # Dissipation: tiny eddies fade out as energy turns to heat
        self.play(
            tiny_eddies.animate.set_alpha(0).scale(1.2),
            rot_tracker.animate.set_value(0.5), # Slow down
            run_time=3
        )
        self.wait(2)
