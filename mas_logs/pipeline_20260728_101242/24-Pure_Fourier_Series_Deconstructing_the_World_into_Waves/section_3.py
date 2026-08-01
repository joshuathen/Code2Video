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

class Section3Scene(TeachingScene):
    def construct(self):
        title = "The Fourier Formula: The Ingredient List"
        lecture_lines = [
            "The Fourier Series formula represents periodic functions mathematically.",
            "Constant a-zero represents the average value or DC offset.",
            "Coefficients a-n and b-n control the amplitude of waves.",
            "Higher frequencies add more detail to the complex signal.",
            "Together, they reconstruct the original function perfectly."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors
        COLOR_FORMULA = WHITE
        COLOR_A0 = "#FFD700"  # Gold
        COLOR_AN_BN = "#ADFF2F"  # GreenYellow
        COLOR_WAVE_1 = "#00FFFF"  # Cyan
        COLOR_WAVE_2 = "#00FF00"  # Green
        COLOR_FINAL = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # The Fourier Series formula represents periodic functions mathematically.
        self.lecture[0].set_color(COLOR_FORMULA)
        
        formula = MathTex(
            r"f(x) =", r"\frac{a_0}{2}", r"+", r"\sum_{n=1}^{\infty}", 
            r"\left[", r"a_n", r"\cos(n\omega x)", r"+", r"b_n", r"\sin(n\omega x)", r"\right]",
            color=COLOR_FORMULA
        )
        # Fix: Issue 26 - Adjust placement and scale
        self.place_in_area(formula, "A2", "B6", scale_factor=0.65)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Constant a-zero represents the average value or DC offset.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_A0)
        
        axes = Axes(
            x_range=[0, 4 * PI, PI],
            y_range=[-2, 6, 1],
            x_length=5,
            y_length=3.5,
            axis_config={"include_tip": False}
        )
        # Fix: Issue 27 - Adjust axes placement and scale
        self.place_in_area(axes, "C1", "F6", scale_factor=0.8)
        
        a0_tracker = ValueTracker(0)
        # Persistent wave mobject using ValueTracker for animation
        wave = axes.plot(lambda x: 1.5 * np.sin(x), color=COLOR_WAVE_1)
        wave.add_updater(lambda m: m.become(axes.plot(
            lambda x: a0_tracker.get_value() + 1.5 * np.sin(x),
            color=COLOR_WAVE_1
        )))
        
        self.play(Create(axes), Create(wave))
        # Highlight a0 in formula
        self.play(formula[1].animate.set_color(COLOR_A0))
        # Shift wave vertically
        self.play(a0_tracker.animate.set_value(2.5), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Coefficients a-n and b-n control the amplitude of waves.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_AN_BN)
        
        an_tracker = ValueTracker(1.5)
        # Update wave to include an_tracker for amplitude scaling
        wave.clear_updaters()
        wave.add_updater(lambda m: m.become(axes.plot(
            lambda x: a0_tracker.get_value() + an_tracker.get_value() * np.sin(x),
            color=COLOR_WAVE_1
        )))
        
        self.play(
            Indicate(formula[5], color=COLOR_AN_BN),
            Indicate(formula[8], color=COLOR_AN_BN),
            formula[5].animate.set_color(COLOR_AN_BN),
            formula[8].animate.set_color(COLOR_AN_BN)
        )
        
        self.play(an_tracker.animate.set_value(0.5), run_time=1)
        self.play(an_tracker.animate.set_value(2.0), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Higher frequencies add more detail to the complex signal.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_WAVE_2)
        
        # Second, faster wave
        n_tracker = ValueTracker(2)
        wave2 = VMobject()
        wave2.add_updater(lambda m: m.become(axes.plot(
            lambda x: 0.8 * np.sin(n_tracker.get_value() * x),
            color=COLOR_WAVE_2
        )))
        
        self.play(Create(wave2))
        self.play(n_tracker.animate.set_value(4), run_time=2)
        
        # Combined signal representation
        combined_wave_obj = axes.plot(
            lambda x: a0_tracker.get_value() + an_tracker.get_value() * np.sin(x) + 0.8 * np.sin(4 * x),
            color=COLOR_FINAL
        )
        
        wave.clear_updaters()
        wave2.clear_updaters()
        self.play(
            FadeOut(wave2),
            ReplacementTransform(wave, combined_wave_obj)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Together, they reconstruct the original function perfectly.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_FINAL)
        
        # Fix: Issue 20 - Integration of Asset for mixing fader
        # Placing fader in B1-A1 range to avoid overlapping with larger axes
        fader_track = Line(self.grid["B1"], self.grid["A1"], color=GRAY)
        fader_knob = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fader.svg")
        fader_knob.scale(0.15).set_color(WHITE)
        fader_knob.move_to(self.grid["B1"])
        fader_label = Text("a1 Level", font_size=16).next_to(fader_track, DOWN, buff=0.1)
        
        fader_group = VGroup(fader_track, fader_knob, fader_label)
        self.play(FadeIn(fader_group))
        
        # Square wave approximation (sum of harmonics)
        def fourier_square(x, n_terms):
            val = 0
            for i in range(1, n_terms + 1, 2):
                val += (4 / (PI * i)) * np.sin(i * x)
            return val

        n_terms_tracker = ValueTracker(1)
        final_wave = VMobject()
        final_wave.add_updater(lambda m: m.become(axes.plot(
            lambda x: 2.5 + fourier_square(x, int(n_terms_tracker.get_value())),
            color=COLOR_FINAL
        )))
        
        self.play(ReplacementTransform(combined_wave_obj, final_wave))
        # Move fader while adding harmonics
        self.play(
            fader_knob.animate.move_to(self.grid["A1"]),
            n_terms_tracker.animate.set_value(13),
            run_time=4,
            rate_func=linear
        )
        self.wait(2)
