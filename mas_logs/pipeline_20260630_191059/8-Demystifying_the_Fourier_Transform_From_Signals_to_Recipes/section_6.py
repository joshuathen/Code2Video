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
        # 1. Setup layout
        title = "The Frequency Domain (The Result)"
        lines = [
            "We plot these center-of-mass peaks on a new graph.",
            "This is the \"Frequency Domain\" view of our signal.",
            "Every peak represents one pure sine wave ingredient.",
            "The horizontal position tells us the ingredient's frequency.",
            "We have successfully decoded the signal's original recipe."
        ]
        self.setup_layout(title, lines)

        # Colors for highlights
        colors = [RED, YELLOW, GREEN, BLUE, PURPLE]

        # === Animation for Lecture Line 1 ===
        # Fade out the winding machine and fade in a horizontal frequency axis.
        self.play(self.lecture[0].animate.set_color(colors[0]))
        
        # Placeholder for previous winding machine (Fix for Issue 59: Moved to C2-E5)
        winding_machine = VGroup(
            Circle(radius=0.7, color=GREY, stroke_width=2, fill_opacity=0.2),
            Line(LEFT*0.5, RIGHT*0.5, color=GREY),
            Line(UP*0.5, DOWN*0.5, color=GREY)
        )
        self.place_in_area(winding_machine, "C2", "E5")
        self.add(winding_machine)
        
        # Frequency Axis (placed along row D)
        axis = Arrow(start=self.grid['D1'], end=self.grid['D6'], color=WHITE, buff=0)
        axis_label = Text("Frequency", font_size=20, color=WHITE)
        self.place_in_area(axis_label, "F2", "F5")
        
        self.play(
            FadeOut(winding_machine),
            Create(axis), 
            Write(axis_label),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Label the axis 'Frequency Domain' with white text.
        # (Fix for Issue 57: Positioned at A3-A5 with scale 0.7)
        self.play(self.lecture[1].animate.set_color(colors[1]))
        
        domain_label = Text("Frequency Domain", color=WHITE, font_size=24)
        self.place_in_area(domain_label, "A3", "A5", scale_factor=0.7)
        
        self.play(FadeIn(domain_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Three vertical spikes (Red, Green, Blue) grow at specific points on the axis.
        self.play(self.lecture[2].animate.set_color(colors[2]))
        
        # Spikes: Red, Green, Blue
        # (Spikes end at row C to allow icons at row B)
        spike_r = Line(self.grid['D2'], self.grid['C2'], color=RED, stroke_width=8)
        spike_g = Line(self.grid['D4'], self.grid['C4'], color=GREEN, stroke_width=8)
        spike_b = Line(self.grid['D6'], self.grid['C6'], color=BLUE, stroke_width=8)
        
        self.play(
            GrowFromEdge(spike_r, DOWN),
            GrowFromEdge(spike_g, DOWN),
            GrowFromEdge(spike_b, DOWN),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Show the original fruit icons (Banana, Blueberry) above their respective spikes.
        # (Fix for Issue 58: Fruit1 at B2, scale 0.8; aligned with others)
        self.play(self.lecture[3].animate.set_color(colors[3]))
        
        # Asset Integration (Issue 46)
        # Using [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/banana.svg]
        try:
            fruit2 = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/banana.svg")
            fruit2.set_color(GREEN)
        except:
            fruit2 = Text("Banana", color=GREEN, font_size=18)
            
        # Strawberry and Blueberry as placeholders/text since no assets provided
        fruit1 = Text("Strawberry", color=RED, font_size=18)
        fruit3 = Text("Blueberry", color=BLUE, font_size=18)
        
        self.place_at_grid(fruit1, "B2", scale_factor=0.8)
        self.place_at_grid(fruit2, "B4", scale_factor=0.8)
        self.place_at_grid(fruit3, "B6", scale_factor=0.8)

        # Frequency values (2Hz, 5Hz, 8Hz) on row E
        freq1 = Text("2 Hz", font_size=16, color=RED)
        freq2 = Text("5 Hz", font_size=16, color=GREEN)
        freq3 = Text("8 Hz", font_size=16, color=BLUE)
        self.place_at_grid(freq1, "E2")
        self.place_at_grid(freq2, "E4")
        self.place_at_grid(freq3, "E6")

        self.play(
            FadeIn(fruit1), FadeIn(fruit2), FadeIn(fruit3),
            Write(freq1), Write(freq2), Write(freq3)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The spikes pulse to emphasize the final 'recipe' result.
        self.play(self.lecture[4].animate.set_color(colors[4]))
        
        pulse_r = spike_r.animate.scale(1.2, about_point=self.grid['D2'])
        pulse_g = spike_g.animate.scale(1.2, about_point=self.grid['D4'])
        pulse_b = spike_b.animate.scale(1.2, about_point=self.grid['D6'])
        
        for _ in range(2):
            self.play(pulse_r, pulse_g, pulse_b, rate_func=there_and_back, run_time=0.8)
        
        self.wait(3)
