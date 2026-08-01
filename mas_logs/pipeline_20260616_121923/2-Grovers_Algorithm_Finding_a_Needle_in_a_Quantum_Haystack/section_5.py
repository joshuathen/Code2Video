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
        # Setup constants and titles
        lecture_lines = [
            'One iteration consists of the Oracle and diffusion.',
            'We repeat this process roughly square root of N times.',
            'This maximizes the probability of finding the correct answer.',
            'Finally, measuring the system collapses it to one state.',
            'We find the golden fish with nearly total certainty.'
        ]
        self.setup_layout("Iteration and Measurement", lecture_lines)
        
        GOLDEN_FISH_COLOR = "#FFD700"
        WHITE_BAR_COLOR = "#FFFFFF"
        HIGHLIGHT_COLOR = YELLOW

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        # Create Bar Chart Elements (7 white bars, 1 gold bar)
        bar_heights = [0.2] * 8
        bars = VGroup(*[
            Rectangle(width=0.25, height=h, fill_opacity=0.8, stroke_width=1) 
            for h in bar_heights
        ]).arrange(RIGHT, buff=0.1)
        
        # Mark the 6th bar as the golden target
        for i, bar in enumerate(bars):
            if i == 5:
                bar.set_color(GOLDEN_FISH_COLOR)
            else:
                bar.set_color(WHITE_BAR_COLOR)
        
        self.place_in_area(bars, "C2", "F5", scale_factor=1.0)
        
        # Label for the iteration component
        iter_comp_label = Text("Oracle + Diffusion", font_size=20, color=WHITE)
        # Fix Issue 39: Move to B2 to avoid obstruction by growing probability bar
        self.place_at_grid(iter_comp_label, "B2", scale_factor=1.0)
        
        self.play(FadeIn(bars), Write(iter_comp_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        # Iteration Counter
        counter_label = Text("sqrt(N) Iterations: ", font_size=24, color=WHITE)
        # Fix Issue 40: Move to A2 to avoid obstruction by growing probability bar
        self.place_at_grid(counter_label, "A2", scale_factor=1.0)
        
        iter_count = ValueTracker(1)
        # Replaced Integer with Text to avoid 'latex' FileNotFoundError
        counter_num = Text("1", font_size=24).next_to(counter_label, RIGHT)
        # Update text value using become() to stay within Text engine
        counter_num.add_updater(lambda m: m.become(
            Text(str(int(iter_count.get_value())), font_size=24).next_to(counter_label, RIGHT)
        ))
        
        self.play(Write(counter_label), FadeIn(counter_num))
        
        # Animate the bars growing (target bar at index 5)
        target_bar = bars[5]
        other_bars = VGroup(*[bars[i] for i in range(len(bars)) if i != 5])
        
        self.play(
            iter_count.animate.set_value(32),
            target_bar.animate.stretch_to_fit_height(3.5).move_to(target_bar.get_bottom(), aligned_edge=DOWN),
            other_bars.animate.stretch_to_fit_height(0.05).move_to(other_bars.get_bottom(), aligned_edge=DOWN),
            run_time=4,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        self.play(
            target_bar.animate.stretch_to_fit_height(5.0).move_to(target_bar.get_bottom(), aligned_edge=DOWN),
            other_bars.animate.stretch_to_fit_height(0.01).move_to(other_bars.get_bottom(), aligned_edge=DOWN),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(HIGHLIGHT_COLOR)
        
        measurement_label = Text("Measurement", font_size=32, color=RED)
        # Fix Issue 38: Move to A5 to avoid horizontal overlap with iter_comp_label at B2/A2
        self.place_at_grid(measurement_label, "A5", scale_factor=1.0)
        
        self.play(
            Write(measurement_label),
            FadeOut(other_bars),
            target_bar.animate.stretch_to_fit_height(5.5).move_to(self.grid["D3"]),
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(HIGHLIGHT_COLOR)
        
        # Replaced MathTex with Text to avoid 'latex' FileNotFoundError
        final_state = Text("|101>", font_size=48, color=GOLDEN_FISH_COLOR)
        self.place_in_area(final_state, "C2", "F5", scale_factor=1.8)
        
        self.play(
            ReplacementTransform(target_bar, final_state),
            FadeOut(iter_comp_label),
            FadeOut(counter_label),
            FadeOut(counter_num),
            FadeOut(measurement_label)
        )
        self.wait(2)
        
        # Cleanup
        self.play(FadeOut(final_state), self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
