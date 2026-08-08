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

class Section2Scene(TeachingScene):
    def construct(self):
        # Setup title and lecture lines
        title_text = "Prerequisite: The Dance of Waves (Interference & Phase)"
        lecture_lines = [
            "Waves interact through a process called interference.",
            "Constructive interference occurs when wave crests align perfectly.",
            "Destructive interference happens when crests meet wave troughs."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors from storyboard
        color_line1 = "#00FFFF"
        color_line2 = "#00FF00"
        color_line3 = "#FF00FF"
        color_ripples = "#FFFFFF"
        color_pattern = "#FFFF00"

        # Utility function for sine waves
        def get_wave(phase, color=color_line1, x_range=[0, 3.5]):
            return FunctionGraph(
                lambda x: 0.5 * np.sin(2 * PI * (x - phase)),
                x_range=x_range,
                color=color
            )

        # === Animation for Lecture Line 1 ===
        # Display a single moving sine wave. Highlight a peak and label it 'Phase' (timing).
        # Also incorporate the pebble/interference example as requested in the outline.
        self.lecture[0].set_color(color_line1)
        
        phase_tracker = ValueTracker(0)
        wave_center = (self.grid["B3"] + self.grid["D6"]) / 2
        
        # Create wave with updater for motion
        wave1 = get_wave(0)
        wave1.move_to(wave_center)
        wave1.add_updater(lambda m: m.become(get_wave(phase_tracker.get_value()).move_to(wave_center)))
        
        phase_label = Text("Phase (timing)", font_size=18, color=color_line1)
        # Update label position to follow a peak
        phase_label.add_updater(lambda m: m.move_to(wave1.input_to_graph_point((phase_tracker.get_value() + 0.25) % 3.5) + UP * 0.4))

        self.play(Create(wave1), FadeIn(phase_label))
        self.play(phase_tracker.animate.set_value(2), run_time=3, rate_func=linear)
        self.wait(1)

        # Transition to Pebble Example (outlined but extra in storyboard steps)
        self.play(FadeOut(wave1), FadeOut(phase_label))
        
        p1_pos = self.grid["C3"]
        p2_pos = self.grid["C5"]
        ripple1 = Circle(radius=0.1, color=color_ripples).move_to(p1_pos)
        ripple2 = Circle(radius=0.1, color=color_ripples).move_to(p2_pos)
        
        self.play(FadeIn(ripple1), FadeIn(ripple2))
        self.play(
            ripple1.animate.scale(15).set_stroke(opacity=0),
            ripple2.animate.scale(15).set_stroke(opacity=0),
            run_time=2
        )
        
        # Interference Pattern spots #FFFF00
        pattern = VGroup()
        for r in ["B", "C", "D", "E"]:
            for c in ["3", "4", "5", "6"]:
                dot = Dot(color=color_pattern, radius=0.08).move_to(self.grid[f"{r}{c}"])
                dist = np.linalg.norm(self.grid[f"{r}{c}"] - (p1_pos + p2_pos)/2)
                dot.set_opacity(np.abs(np.sin(dist * 3)))
                pattern.add(dot)
        
        self.play(FadeIn(pattern))
        self.wait(2)
        self.play(FadeOut(pattern), FadeOut(ripple1), FadeOut(ripple2))
        self.lecture[0].set_color(WHITE)

        # === Animation for Lecture Line 2 ===
        # Show two sine waves overlapping perfectly. Their sum grows into a larger wave (Constructive Interference).
        self.lecture[1].set_color(color_line2)
        
        w2a = get_wave(0, color=color_line2, x_range=[0, 3]).shift(UP * 0.4)
        w2b = get_wave(0, color=color_line2, x_range=[0, 3]).shift(DOWN * 0.4)
        v_const = VGroup(w2a, w2b)
        self.place_in_area(v_const, "B3", "C6")
        
        sum_wave = FunctionGraph(
            lambda x: 1.0 * np.sin(2 * PI * x),
            x_range=[0, 3],
            color=color_line2
        )
        self.place_in_area(sum_wave, "D3", "E6")
        
        label_const = Text("Constructive", font_size=20, color=color_line2)
        self.place_at_grid(label_const, "A4")

        self.play(Create(w2a), Create(w2b), Write(label_const))
        self.wait(1)
        self.play(
            w2a.animate.move_to(sum_wave.get_center()),
            w2b.animate.move_to(sum_wave.get_center())
        )
        self.play(ReplacementTransform(VGroup(w2a, w2b), sum_wave))
        self.wait(2)
        self.play(FadeOut(sum_wave), FadeOut(label_const))
        self.lecture[1].set_color(WHITE)

        # === Animation for Lecture Line 3 ===
        # Shift one wave so peaks meet troughs. Their sum becomes a flat line (Destructive Interference).
        self.lecture[2].set_color(color_line3)
        
        w3a = get_wave(0, color=color_line3, x_range=[0, 3]).shift(UP * 0.4)
        w3b = get_wave(0.5, color=color_line3, x_range=[0, 3]).shift(DOWN * 0.4) # phase 0.5 = 180 degrees
        v_dest = VGroup(w3a, w3b)
        self.place_in_area(v_dest, "B3", "C6")
        
        flat_line = Line(
            self.grid["D3"], self.grid["D6"],
            color=color_line3, stroke_width=4
        )
        self.place_in_area(flat_line, "D3", "E6")
        
        label_dest = Text("Destructive", font_size=20, color=color_line3)
        self.place_at_grid(label_dest, "A4")

        self.play(Create(w3a), Create(w3b), Write(label_dest))
        self.wait(1)
        self.play(
            w3a.animate.move_to(flat_line.get_center()),
            w3b.animate.move_to(flat_line.get_center())
        )
        self.play(ReplacementTransform(VGroup(w3a, w3b), flat_line))
        self.wait(2)
        self.play(FadeOut(flat_line), FadeOut(label_dest))
        self.lecture[2].set_color(WHITE)
