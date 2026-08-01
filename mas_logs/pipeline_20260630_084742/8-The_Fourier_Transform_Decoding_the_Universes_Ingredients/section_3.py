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
        # Define Colors
        CYAN = "#00FFFF"
        RED = "#FF0000"
        GREEN = "#00FF00"
        BLUE = "#0000FF"
        YELLOW = "#FFFF00"
        
        lecture_lines = [
            'We usually see signals as amplitude over time.', 
            'This is known as the Time Domain view.', 
            'But we can shift to the Frequency Domain.', 
            'Waves become clean bars at specific frequencies.', 
            'The Fourier Transform bridges these two worlds.'
        ]
        
        self.setup_layout("A Change of Perspective: Time vs. Frequency", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(CYAN)
        
        # Create Time Domain Axes - Issue 40: Moved to B2-E6
        time_axes = Axes(
            x_range=[0, 4, 1],
            y_range=[-2, 2, 1],
            x_length=4.5,
            y_length=3,
            axis_config={"include_tip": True, "color": WHITE},
            tips=False
        )
        self.place_in_area(time_axes, "B2", "E6")
        
        # Complex wave: sum of 3 sines
        wave = time_axes.plot(
            lambda t: 0.8*np.sin(2*np.pi*1*t) + 0.5*np.sin(2*np.pi*2*t) + 0.3*np.sin(2*np.pi*3*t),
            color=CYAN
        )
        
        # Issue 42: scaled to 0.7
        time_label = Text("Time", font_size=18, color=WHITE)
        self.place_at_grid(time_label, "F4", scale_factor=0.7)
        
        # Issue 41: scaled to 0.7
        amp_label = Text("Amplitude", font_size=18, color=WHITE).rotate(90*DEGREES)
        self.place_at_grid(amp_label, "C1", scale_factor=0.7)
        
        # Requirement: Display text 'Time Domain' above the wave
        time_domain_title = Text("Time Domain", font_size=20, color=WHITE)
        self.place_at_grid(time_domain_title, "A4")
        
        self.play(Create(time_axes), Create(wave), Write(time_label), Write(amp_label), Write(time_domain_title), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(CYAN)
        self.play(Indicate(time_domain_title), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Create Frequency Domain Axes - Issue 40: Moved to B2-E6
        freq_axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 1.5, 0.5],
            x_length=4.5,
            y_length=3,
            axis_config={"include_tip": True, "color": WHITE},
            tips=False
        )
        self.place_in_area(freq_axes, "B2", "E6")
        
        # Issue 42: scaled to 0.7
        freq_axis_label = Text("Frequency", font_size=18, color=WHITE)
        self.place_at_grid(freq_axis_label, "F4", scale_factor=0.7)
        
        # Issue 41: scaled to 0.7
        strength_label = Text("Strength", font_size=18, color=WHITE).rotate(90*DEGREES)
        self.place_at_grid(strength_label, "C1", scale_factor=0.7)
        
        freq_domain_title = Text("Frequency Domain", font_size=20, color=WHITE)
        self.place_at_grid(freq_domain_title, "A4")
        
        self.play(
            FadeOut(wave),
            ReplacementTransform(time_axes, freq_axes),
            ReplacementTransform(time_label, freq_axis_label),
            ReplacementTransform(amp_label, strength_label),
            ReplacementTransform(time_domain_title, freq_domain_title),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Match colors: Red for f=1, Green for f=2, Blue for f=3
        self.lecture[3].set_color(YELLOW)
        
        # Bars representing spectral peaks
        bar1 = Line(freq_axes.c2p(1, 0), freq_axes.c2p(1, 0.8), color=RED, stroke_width=8)
        bar2 = Line(freq_axes.c2p(2, 0), freq_axes.c2p(2, 0.5), color=GREEN, stroke_width=8)
        bar3 = Line(freq_axes.c2p(3, 0), freq_axes.c2p(3, 0.3), color=BLUE, stroke_width=8)
        
        dot1 = Dot(freq_axes.c2p(1, 0.8), color=RED)
        dot2 = Dot(freq_axes.c2p(2, 0.5), color=GREEN)
        dot3 = Dot(freq_axes.c2p(3, 0.3), color=BLUE)
        
        impulses = VGroup(bar1, dot1, bar2, dot2, bar3, dot3)
        
        self.play(Create(impulses), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Bridge icon connecting 'Time' and 'Frequency' concepts
        bridge_label_time = Text("Time", font_size=14, color=WHITE)
        self.place_at_grid(bridge_label_time, "F2")
        bridge_label_freq = Text("Frequency", font_size=14, color=WHITE)
        self.place_at_grid(bridge_label_freq, "F6")
        
        # Draw a glowing bridge (arc)
        bridge_arc = ArcBetweenPoints(
            self.grid["F2"] + RIGHT*0.4, 
            self.grid["F6"] + LEFT*0.4, 
            angle=-TAU/4, 
            color=WHITE
        )
        bridge_glow = bridge_arc.copy().set_stroke(WHITE, opacity=0.4, width=8)
        bridge_icon = VGroup(bridge_glow, bridge_arc)
        
        self.play(Write(bridge_label_time), Write(bridge_label_freq), Create(bridge_icon), run_time=2)
        self.play(Indicate(freq_axes), Indicate(impulses), run_time=2)
        self.wait(2)
