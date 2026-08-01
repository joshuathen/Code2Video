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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup layout with exactly 5 lines as per script
        title = "Visual Synthesis: Primes Constructing a Circle"
        lines = [
            "The straight lanes of arithmetic now bend and curve.",
            "They join to form the perfect geometry of a circle.",
            "A sweeping radius measures this newfound circular unity.",
            "Pi emerges as the hidden architect of prime distribution.",
            "Discrete numbers finally reveal their smooth, eternal nature."
        ]
        self.setup_layout(title, lines)

        # Define visual center reference point (using grid)
        # Main geometry area B2-E5
        center_anchor = Dot(radius=0)
        self.place_in_area(center_anchor, "B2", "E5")
        center_pos = center_anchor.get_center()

        # === Animation for Lecture Line 1 ===
        # The straight lanes of arithmetic now bend and curve.
        self.lecture[0].set_color(YELLOW)
        
        # Create two straight horizontal paths
        lane_top = Line(self.grid["B2"], self.grid["B5"], color=WHITE)
        lane_bottom = Line(self.grid["E2"], self.grid["E5"], color=WHITE)
        
        self.play(Create(lane_top), Create(lane_bottom))
        self.wait(0.5)

        # Transition lanes to bending arcs
        # Arcs centered at center_pos with radius 1.5
        arc_top = Arc(radius=1.5, start_angle=0, angle=PI, color=WHITE).move_to(center_pos)
        arc_bottom = Arc(radius=1.5, start_angle=PI, angle=PI, color=WHITE).move_to(center_pos)
        
        self.play(
            ReplacementTransform(lane_top, arc_top),
            ReplacementTransform(lane_bottom, arc_bottom),
            run_time=2
        )
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # They join to form the perfect geometry of a circle.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Form a full white circle (#FFFFFF)
        full_circle = Circle(radius=1.5, color=WHITE).move_to(center_pos)
        self.play(ReplacementTransform(VGroup(arc_top, arc_bottom), full_circle))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # A sweeping radius measures this newfound circular unity.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Red radius vector (#FF0000)
        radius_vector = Line(center_pos, center_pos + RIGHT * 1.5, color="#FF0000", stroke_width=4)
        radius_label = Text("r", color="#FF0000", slant=ITALIC).scale(0.8)
        
        # Updater for labeling the radius
        def update_label(label):
            unit_vec = radius_vector.get_unit_vector()
            label.move_to(radius_vector.get_end() + unit_vec * 0.3)
        
        radius_label.add_updater(update_label)
        self.play(Create(radius_vector), Write(radius_label))
        
        # Sweep radius vector 360 degrees
        angle_tracker = ValueTracker(0)
        radius_vector.add_updater(
            lambda m: m.put_start_and_end_on(
                center_pos, 
                center_pos + 1.5 * np.array([np.cos(angle_tracker.get_value()), np.sin(angle_tracker.get_value()), 0])
            )
        )
        
        self.play(angle_tracker.animate.set_value(2 * PI), run_time=3, rate_func=linear)
        self.wait(1)
        radius_label.remove_updater(update_label)

        # === Animation for Lecture Line 4 ===
        # Pi emerges as the hidden architect of prime distribution.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)

        # Flash text "Primes Construct Pi" in white (#FFFFFF)
        # Resolved Issue 32: Positioning at A2-B5 to avoid overlapping center geometry
        flash_text = Text("Primes Construct Pi", font_size=32, color=WHITE)
        self.place_in_area(flash_text, "A2", "B5", scale_factor=0.7)
        
        self.play(FadeIn(flash_text))
        self.play(Indicate(flash_text, color=WHITE, scale_factor=1.1))
        self.wait(1)
        self.play(FadeOut(flash_text))

        # === Animation for Lecture Line 5 ===
        # Discrete numbers finally reveal their smooth, eternal nature.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)

        # Circle fades, leaving a glowing golden (#FFD700) Pi symbol
        pi_symbol = Text("π", color="#FFD700").scale(4)
        self.place_in_area(pi_symbol, "B2", "E5")
        
        pi_glow = Text("π", color="#FFD700").scale(4.2).set_opacity(0.3)
        self.place_in_area(pi_glow, "B2", "E5")

        self.play(
            FadeOut(full_circle),
            FadeOut(radius_vector),
            FadeOut(radius_label),
            FadeIn(pi_symbol),
            FadeIn(pi_glow),
            run_time=2
        )
        
        # Final subtle pulse effect for the Pi symbol
        self.play(
            pi_symbol.animate.scale(1.1),
            pi_glow.animate.scale(1.2).set_opacity(0.1),
            rate_func=there_and_back,
            run_time=2
        )
        
        self.wait(2)
