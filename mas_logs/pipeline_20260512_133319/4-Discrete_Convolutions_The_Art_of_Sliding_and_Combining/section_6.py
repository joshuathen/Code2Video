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
        # Initialize Scene
        lecture_lines = [
            'Convolution is essentially a weighted local average.',
            'It powers modern breakthroughs in Convolutional Neural Networks.',
            'From signals to AI, convolution extracts deep meaning.'
        ]
        self.setup_layout("Summary: The Universal Tool", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line to match Cyan icons
        self.play(self.lecture[0].animate.set_color("#00FFFF"), run_time=0.5)

        # Central Kernel
        kernel_box = Square(side_length=1.0, color="#00FFFF", stroke_width=2)
        kernel_text = Text("Kernel", font_size=20, color="#00FFFF")
        kernel_group = VGroup(kernel_box, kernel_text)
        self.place_at_grid(kernel_group, "C3", scale_factor=1.0)

        # Audio Icon (Bars)
        audio_bars = VGroup(*[Rectangle(width=0.1, height=h, fill_opacity=1, color="#00FFFF", stroke_width=0) 
                             for h in [0.4, 0.7, 0.5, 0.8, 0.3]]).arrange(RIGHT, buff=0.05)
        audio_label = Text("Audio", font_size=16, color="#00FFFF").next_to(audio_bars, DOWN, buff=0.1)
        audio_icon = VGroup(audio_bars, audio_label)
        self.place_at_grid(audio_icon, "A3", scale_factor=0.8)

        # Imaging Icon (Grid) - Fix Issue 39: Move to C4 for better balance
        imaging_grid = Rectangle(width=0.6, height=0.6, color="#00FFFF").add(
            Line(np.array([-0.3, 0, 0]), np.array([0.3, 0, 0]), color="#00FFFF", stroke_width=1),
            Line(np.array([0, -0.3, 0]), np.array([0, 0.3, 0]), color="#00FFFF", stroke_width=1)
        )
        imaging_label = Text("Imaging", font_size=16, color="#00FFFF").next_to(imaging_grid, DOWN, buff=0.1)
        imaging_icon = VGroup(imaging_grid, imaging_label)
        self.place_at_grid(imaging_icon, "C4", scale_factor=0.8)

        # AI Icon (Simple Graph)
        d1, d2, d3 = Dot(color="#00FFFF"), Dot(color="#00FFFF"), Dot(color="#00FFFF")
        d1.move_to(UP*0.3)
        d2.move_to(LEFT*0.3 + DOWN*0.3)
        d3.move_to(RIGHT*0.3 + DOWN*0.3)
        ai_lines = VGroup(Line(d1, d2, color="#00FFFF"), Line(d1, d3, color="#00FFFF"), Line(d2, d3, color="#00FFFF"))
        ai_label = Text("AI", font_size=16, color="#00FFFF").next_to(ai_lines, DOWN, buff=0.1)
        ai_icon = VGroup(d1, d2, d3, ai_lines, ai_label)
        self.place_at_grid(ai_icon, "E3", scale_factor=0.8)

        # Animation: Icons appear around kernel
        self.play(
            FadeIn(kernel_group),
            FadeIn(audio_icon, shift=DOWN),
            FadeIn(imaging_icon, shift=LEFT),
            FadeIn(ai_icon, shift=UP),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second lecture line, reset previous
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(WHITE),
            run_time=0.5
        )

        # "Weighted Local Average" text
        # Fix Issue 38 & 40: Place at F1-F6 area with scale 0.8 to avoid overlap with central icons
        weighted_avg_text = Text("Weighted Local Average", color="#FFFFFF", font_size=32)
        self.place_in_area(weighted_avg_text, "F1", "F6", scale_factor=0.8)

        # Clear existing icons and bring in the summary text
        self.play(
            FadeOut(audio_icon), FadeOut(imaging_icon), FadeOut(ai_icon), FadeOut(kernel_group),
            FadeIn(weighted_avg_text, shift=UP),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third lecture line
        self.play(self.lecture[2].animate.set_color(WHITE), run_time=0.5)

        # Final Summary Quote
        final_quote = Text("Convolution:\nThe Language of Systems", color="#FFFFFF", font_size=36, line_spacing=1.0)
        
        # Animation: Fade out UI elements to transition to the final message
        self.play(
            FadeOut(weighted_avg_text),
            FadeOut(self.lecture),
            FadeOut(self.title),
            run_time=1.5
        )
        
        # Center final quote on entire screen for the conclusion
        final_quote.move_to(ORIGIN)
        self.play(FadeIn(final_quote), run_time=2)
        self.wait(2)
