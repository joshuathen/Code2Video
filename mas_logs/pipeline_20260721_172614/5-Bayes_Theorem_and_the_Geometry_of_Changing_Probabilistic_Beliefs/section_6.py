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
        title_text = "Summary: Belief as a Dynamic Shape"
        lecture_lines = [
            "Probabilistic belief is a dynamic shape that shifts.",
            "Extraordinary claims require massive evidence to change this space.",
            "Use geometry to master the logic of uncertain worlds."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Assets
        slider_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/slider.svg")
        m_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/m.svg")

        # === Animation for Lecture Line 1 ===
        # Highlight Line 1
        self.play(self.lecture[0].animate.set_color(BLUE_C))

        # Dynamic Shape: A rectangle that morphs
        shape = Rectangle(width=2, height=2, color=BLUE_B, fill_opacity=0.5)
        # Fix Issue 42: Position at C3-E5
        self.place_in_area(shape, "C3", "E5")
        
        # Fix Issue 43: Label at B4
        label_belief = Text("Belief Area", font_size=24, color=BLUE_B)
        self.place_at_grid(label_belief, "B4", scale_factor=0.8)

        self.play(DrawBorderThenFill(shape), Write(label_belief))
        self.wait(0.5)

        # Storyboard Stage 1: "Show a quick replay of the area shrinking and stretching"
        self.play(shape.animate.stretch_to_fit_width(0.5, about_edge=LEFT).stretch_to_fit_height(3), run_time=0.4)
        self.play(shape.animate.stretch_to_fit_width(3, about_edge=LEFT).stretch_to_fit_height(0.5), run_time=0.4)
        self.play(shape.animate.stretch_to_fit_width(2, about_edge=LEFT).stretch_to_fit_height(2), run_time=0.4)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Reset colors and highlight Line 2
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(GREEN_B)
        )

        # Extraordinary claim: Small Prior area
        new_label_prior = Text("Prior (Rare Claim)", font_size=24, color=RED_B)
        # Fix Issue 43: Label at B4
        self.place_at_grid(new_label_prior, "B4", scale_factor=0.8)

        self.play(
            shape.animate.stretch_to_fit_width(0.3, about_edge=LEFT).set_color(RED_B),
            FadeTransform(label_belief, new_label_prior)
        )
        self.wait(0.5)

        # Slider Setup (Storyboard Stage 2)
        # Fix Issue 44: slider_label in area F3-F5
        self.place_in_area(slider_asset, "F2", "F5", scale_factor=0.8)
        # m_icon as label for P(E|~H)
        self.place_at_grid(m_icon, "F2", scale_factor=0.4).shift(LEFT * 0.7)
        
        slider_label = Text("Mound Frequency P(E|~H)", font_size=18, color=WHITE)
        self.place_in_area(slider_label, "F3", "F5", scale_factor=0.8).shift(UP * 0.5)
        
        # Tracker for P(E|~H) - frequency of evidence in the general world
        tracker = ValueTracker(0.8) # Start with high frequency (evidence is common)
        
        # Knob for the slider
        knob = Dot(color=YELLOW)
        def update_knob(k):
            # Map tracker value (0 to 1) to slider x position
            left_x = slider_asset.get_left()[0]
            right_x = slider_asset.get_right()[0]
            y_val = slider_asset.get_center()[1]
            x_val = left_x + tracker.get_value() * (right_x - left_x)
            k.move_to([x_val, y_val, 0])
        
        knob.add_updater(update_knob)

        # Dynamic Belief Updates based on tracker (Storyboard Stage 3)
        def update_shape(s):
            p_h = 0.1
            p_e_h = 0.9
            p_e_not_h = tracker.get_value()
            # Bayes formula for posterior
            numerator = p_e_h * p_h
            denominator = numerator + p_e_not_h * (1 - p_h)
            p_h_e = numerator / denominator
            # Map posterior probability to rectangle width (max width ~4 grid units)
            s.stretch_to_fit_width(0.3 + p_h_e * 4.0, about_edge=LEFT)

        shape.add_updater(update_shape)

        self.add(slider_asset, m_icon, slider_label, knob)
        self.play(FadeIn(slider_asset), FadeIn(m_icon), FadeIn(slider_label), FadeIn(knob))
        self.wait(0.5)

        # Demonstrate: If P(E|~H) is high (0.8), area doesn't change much from prior.
        self.wait(1.5)

        # Move tracker to low (0.05) -> Evidence is rare, belief grows significantly.
        self.play(tracker.animate.set_value(0.05), run_time=2)
        self.wait(1.5)
        
        # Move back to high (0.9) -> Belief shrinks back.
        self.play(tracker.animate.set_value(0.9), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Reset colors and highlight Line 3
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(GOLD_B)
        )

        new_label_posterior = Text("Posterior Belief", font_size=24, color=YELLOW_B)
        self.place_at_grid(new_label_posterior, "B4", scale_factor=0.8)
        
        self.play(FadeTransform(new_label_prior, new_label_posterior))
        self.wait(0.5)
        
        # Final visual: Finalize and scale everything
        shape.clear_updaters()
        knob.clear_updaters()
        
        all_visuals = VGroup(shape, new_label_posterior, slider_asset, slider_label, knob, m_icon)
        self.play(all_visuals.animate.scale(1.05), run_time=1)
        self.wait(2)

        # Fade out everything
        self.play(
            FadeOut(all_visuals),
            FadeOut(self.lecture),
            FadeOut(self.title)
        )
        self.wait(1)
