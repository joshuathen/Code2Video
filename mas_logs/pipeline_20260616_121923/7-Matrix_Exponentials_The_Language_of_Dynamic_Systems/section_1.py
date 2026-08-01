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
        # Setup layout with specific title and lines from Stage 3 prompt
        self.setup_layout(
            "From Numbers to Matrices: The Motivation", 
            [
                "Scalar exponentials solve simple linear differential equations.",
                "The growth factor e^{at} scales the initial state.",
                "However, most systems involve multiple interacting variables.",
                "The matrix exponential evolves these linked states simultaneously.",
                "Digital pet levels depend on each other through matrices."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Step 1: Create y' = ay
        line1_highlight = YELLOW
        self.play(self.lecture[0].animate.set_color(line1_highlight))
        
        eq_scalar = MarkupText("y' = ay", color=WHITE)
        self.place_in_area(eq_scalar, "A3", "A4", scale_factor=1.2)
        self.play(FadeIn(eq_scalar))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # Step 2: Create solution y(t) = e^{at}y(0)
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        sol_scalar = VGroup(
            MarkupText("y(t) = "), 
            MarkupText("e<sup>at</sup>"), 
            MarkupText("y(0)")
        ).arrange(RIGHT, buff=0.1).set_color(WHITE)
        
        sol_scalar[1].set_color(YELLOW)
        self.place_in_area(sol_scalar, "B3", "B4", scale_factor=1.2)
        self.play(FadeIn(sol_scalar))
        self.wait(2)
        
        # === Animation for Lecture Line 3 ===
        # Step 3: Transform into vector notation
        line3_highlight = BLUE_B
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(line3_highlight)
        )
        
        eq_vector = MarkupText("<b>x</b>' = A<b>x</b>", color=WHITE)
        sol_vector = VGroup(
            MarkupText("<b>x</b>(t) = "), 
            MarkupText("e<sup>At</sup>"), 
            MarkupText("<b>x</b>(0)")
        ).arrange(RIGHT, buff=0.1).set_color(WHITE)
        
        self.place_in_area(eq_vector, "A3", "A4", scale_factor=1.2)
        self.place_in_area(sol_vector, "B3", "B4", scale_factor=1.2)
        
        self.play(
            ReplacementTransform(eq_scalar, eq_vector),
            ReplacementTransform(sol_scalar, sol_vector)
        )
        self.wait(1)
        
        # === Animation for Lecture Line 4 ===
        # Step 4: Pulse e^{At} in #00FF00
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color("#00FF00")
        )
        target_term = sol_vector[1]
        self.play(target_term.animate.set_color("#00FF00"))
        self.play(
            target_term.animate.scale(1.3),
            run_time=0.5, rate_func=there_and_back
        )
        self.play(
            target_term.animate.scale(1.3),
            run_time=0.5, rate_func=there_and_back
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Step 5: Digital Pet metaphor (Hunger and Tiredness bars)
        line5_highlight = PINK
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(line5_highlight)
        )
        
        # Fade out old formulas to make room
        self.play(FadeOut(eq_vector), FadeOut(sol_vector))
        
        # Create bars
        hunger_bar = Rectangle(width=0.8, height=1.0, fill_opacity=0.8, fill_color="#FF00FF", stroke_color=WHITE)
        tired_bar = Rectangle(width=0.8, height=1.5, fill_opacity=0.8, fill_color="#0000FF", stroke_color=WHITE)
        
        # Issue 29: hunger_label at B2
        hunger_label = Text("Hunger", font_size=16, color="#FF00FF")
        self.place_at_grid(hunger_label, "B2", scale_factor=1.0)
        
        # Issue 28: tired_label at B4
        tired_label = Text("Tiredness", font_size=16, color="#0000FF")
        self.place_at_grid(tired_label, "B4", scale_factor=1.0)
        
        # Placement for bars
        self.place_at_grid(hunger_bar, "D2", scale_factor=1.0)
        hunger_bar.align_to(self.grid["E2"], DOWN)
        
        # Issue 30: tired_bar at D4
        self.place_at_grid(tired_bar, "D4", scale_factor=1.0)
        tired_bar.align_to(self.grid["E4"], DOWN)
        
        self.play(
            Create(hunger_bar), Create(tired_bar),
            Write(hunger_label), Write(tired_label)
        )
        
        # Evolution values
        time_tracker = ValueTracker(0)
        
        def update_hunger(m):
            t = time_tracker.get_value()
            new_h = 1.0 + 0.3 * t + 0.1 * np.sin(2*t)
            m.stretch_to_fit_height(max(0.1, new_h), about_edge=DOWN)
            
        def update_tired(m):
            t = time_tracker.get_value()
            new_r = 1.5 + 0.2 * t + 0.3 * np.cos(2*t)
            m.stretch_to_fit_height(max(0.1, new_r), about_edge=DOWN)
            
        hunger_bar.add_updater(update_hunger)
        tired_bar.add_updater(update_tired)
        
        self.play(time_tracker.animate.set_value(3), run_time=4, rate_func=linear)
        self.wait(1)
        
        hunger_bar.remove_updater(update_hunger)
        tired_bar.remove_updater(update_tired)
        
        # Reset color
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(2)
