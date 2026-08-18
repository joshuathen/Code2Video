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
        # Define lecture lines based on storyboard
        lecture_lines = [
            "- A continuous signal can be represented as a smooth wave.",
            "- Sampling measures the wave's amplitude at specific discrete intervals.",
            "- These measurements form an array of numbers called a signal."
        ]
        
        # Setup the layout
        self.setup_layout("Prerequisite Knowledge: Signals as Arrays", lecture_lines)
        
        # Visual Colors
        WAVE_COLOR = "#FF00FF"
        DOT_COLOR = "#00FF00"
        BOX_COLOR = "#FFFFFF"

        # === 1. Prepare Mobjects ===
        
        # Smooth Wave Asset
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/wave.svg]
        wave_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wave.svg")
        wave_svg.set_color(WAVE_COLOR)
        
        # Sampling Dots
        # Define sample points along a virtual sine wave for consistent data
        sample_x_values = np.linspace(-2.2, 2.2, 6)
        dots = VGroup(*[
            Dot(point=[x, np.sin(x), 0], color=DOT_COLOR)
            for x in sample_x_values
        ])
        
        # Match sizes and center before grouping to ensure alignment
        wave_svg.stretch_to_fit_width(4).stretch_to_fit_height(1.5).move_to(ORIGIN)
        dots.stretch_to_fit_width(4).stretch_to_fit_height(1.5).move_to(ORIGIN)
        
        # Group and position signal area
        signal_area = VGroup(wave_svg, dots)
        # Fix: Resize signal_area (wave and dots) to 'B1'-'D6' with scale_factor=1.2
        self.place_in_area(signal_area, "B1", "D6", scale_factor=1.2)
        
        # Measurement Array (Boxes and Labels)
        boxes = VGroup()
        labels = VGroup()
        for i, x_val in enumerate(sample_x_values):
            box = Square(side_length=0.8, color=BOX_COLOR)
            
            # Numerical label (amplitude)
            amplitude = np.sin(x_val)
            label = Text(f"{amplitude:.1f}", font_size=18, color=BOX_COLOR)
            label.move_to(box.get_center())
            
            boxes.add(box)
            labels.add(label)

        # Group boxes and labels for consistent transformation
        measurement_array = VGroup()
        for b, l in zip(boxes, labels):
            measurement_array.add(VGroup(b, l))
        
        measurement_array.arrange(RIGHT, buff=0.2)
        # Fix: Move measurement_array (boxes) to 'E1'-'F6' with scale_factor=0.8
        self.place_in_area(measurement_array, "E1", "F6", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        # Draw a smooth sine wave (#FF00FF) across the center.
        self.play(self.lecture[0].animate.set_color(WAVE_COLOR))
        self.play(Create(wave_svg), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Place vertical dots (#00FF00) along the sine wave to show sampling.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(DOT_COLOR)
        )
        self.play(Create(dots), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Organize dots into a horizontal row of boxes (#FFFFFF) containing numerical values.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(BOX_COLOR)
        )
        
        # Fade in boxes (the square frames)
        self.play(FadeIn(boxes, shift=UP), run_time=1)
        
        # Transform copies of dots to the labels inside boxes
        self.play(
            ReplacementTransform(dots.copy(), labels),
            dots.animate.set_opacity(0.3),
            run_time=2
        )
        self.wait(3)
