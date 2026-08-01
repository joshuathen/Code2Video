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
        # Define the lecture content
        title = "Step 1: The Oracle (Phase Inversion)"
        lines = [
            "The Oracle function marks the correct answer.",
            "It flips the target state's amplitude to negative.",
            "Other amplitudes remain positive and unchanged.",
            "The target is now mathematically distinct from others.",
            "This phase inversion prepares for the next step."
        ]
        self.setup_layout(title, lines)
        
        # Colors
        COLOR_TARGET = "#FFFF00"  # Yellow
        COLOR_OTHER = "#FFFFFF"   # White

        # === Animation for Lecture Line 1 ===
        # The Oracle function marks the correct answer.
        self.lecture[0].set_color(COLOR_TARGET)
        
        # Create 8 bars representing the amplitudes of 3 qubits
        bars = VGroup(*[
            Rectangle(
                width=0.3, 
                height=1.5, 
                fill_opacity=1, 
                fill_color=COLOR_OTHER, 
                stroke_width=1, 
                stroke_color=COLOR_OTHER
            ) for _ in range(8)
        ])
        bars.arrange(RIGHT, buff=0.2)
        # Highlight the 6th bar (index 5) for state |101>
        bars[5].set_fill(COLOR_TARGET)
        bars[5].set_stroke(COLOR_TARGET)
        
        # Labels for the states
        labels = VGroup(*[
            Text(f"|{i:03b}⟩", font_size=18, color=WHITE) 
            for i in range(8)
        ])
        for i, label in enumerate(labels):
            label.next_to(bars[i], DOWN, buff=0.1)
            
        chart = VGroup(bars, labels)
        # Position the chart in the grid area - adjusted based on VideoCritic feedback
        # Prevents cutoff and improves readability by centering vertically in rows A-E
        self.place_in_area(chart, "A2", "E6", scale_factor=0.8)
        
        # Oracle target icon - Integrated asset [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/target.svg]
        target_icon = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/target.svg", color=COLOR_TARGET, height=0.4)
        target_icon.next_to(bars[5], UP, buff=0.2)
        
        self.play(FadeIn(chart))
        self.play(Create(target_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # It flips the target state's amplitude to negative.
        self.lecture[1].set_color(COLOR_TARGET)
        
        # Perform the phase inversion (flip the bar across the x-axis)
        # We rotate around the bottom edge of the bar
        self.play(
            Rotate(
                bars[5], 
                angle=PI, 
                axis=RIGHT, 
                about_point=bars[5].get_bottom()
            ),
            labels[5].animate.shift(DOWN * 1.5), # Shift label down to stay below the flipped bar
            FadeOut(target_icon, scale=0.5), # Icon disappears as it marks the completion of marking
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Other amplitudes remain positive and unchanged.
        self.lecture[2].set_color(COLOR_OTHER)
        
        # Briefly highlight the unchanged bars to draw attention to them
        other_bars = VGroup(*[bars[i] for i in range(8) if i != 5])
        self.play(
            other_bars.animate.set_opacity(0.5),
            run_time=0.5
        )
        self.play(
            other_bars.animate.set_opacity(1.0),
            run_time=0.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The target is now mathematically distinct from others.
        self.lecture[3].set_color(COLOR_TARGET)
        
        # Add yellow negative sign next to the inverted bar
        neg_sign = Text("-", color=COLOR_TARGET, font_size=36)
        neg_sign.next_to(bars[5], LEFT, buff=0.1)
        
        self.play(FadeIn(neg_sign))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This phase inversion prepares for the next step.
        self.lecture[4].set_color(COLOR_OTHER)
        
        # Final highlight of the result
        self.play(Indicate(VGroup(bars[5], neg_sign), color=COLOR_TARGET))
        self.wait(2)
