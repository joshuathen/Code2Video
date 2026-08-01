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
        # --- Data Setup ---
        title_text = "The Core Concept: Updating Beliefs"
        lecture_lines = [
            "Probability is your degree of belief in an event.",
            "New evidence should change how you view the world.",
            "This update is the heart of Bayes' Theorem."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors (L008: Hexadecimal strings)
        color1 = "#ADD8E6"  # Light Blue
        color2 = "#32CD32"  # Lime Green
        color3 = "#FF4500"  # Orange Red

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line
        self.play(self.lecture[0].animate.set_color(color1))
        self.wait(1.5)

        # Display 'Probability: Your Degree of Belief' in #ADD8E6.
        # Issue 29: Place in area A2-A5 to avoid vertical crowding.
        belief_text = Text("Probability: Your Degree of Belief", font_size=24, color=color1)
        self.place_in_area(belief_text, "A2", "A5", scale_factor=0.7)
        self.play(Write(belief_text))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(color2)
        )
        self.wait(1.5)

        # Create a meter [Asset: meter.svg] showing a '20% Chance of Rain' (#32CD32).
        # Issue 22: Integrate Asset /scratch/pawsey1357/jthen/Code2Video/assets/icon/meter.svg
        meter_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/meter.svg")
        meter_svg.set_height(0.6)
        
        # Create a fill bar that represents 20%
        # L025: Use set_height/width without stretch or use scale.
        fill_width_20 = 1.5 * 0.2
        meter_fill = Rectangle(
            width=fill_width_20, 
            height=0.4, 
            stroke_width=0
        ).set_fill(color=color2, opacity=1.0) # L031: use set_fill for opacity
        
        # Align fill with SVG. Assuming SVG is roughly horizontal bar-like.
        meter_svg.set_width(1.5)
        meter_fill.move_to(meter_svg.get_left(), aligned_edge=LEFT).shift(RIGHT * 0.1)
        
        meter_group = VGroup(meter_svg, meter_fill)
        # Issue 28: Move meter_group to E3.
        self.place_at_grid(meter_group, "E3", scale_factor=0.8)
        
        rain_label = Text("20% Chance of Rain", font_size=22, color=color2)
        self.place_at_grid(rain_label, "C3", scale_factor=0.8)
        
        self.play(FadeIn(meter_svg), FadeIn(meter_fill), Write(rain_label))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(color3)
        )
        self.wait(1.5)

        # Update the meter [Asset: meter.svg] to 69% as dark clouds appear (#FF4500).
        # Issue 27: Move cloud_circles to C6.
        cloud_circles = VGroup(
            Circle(radius=0.3).set_fill(color="#808080", opacity=1.0).set_stroke(width=0),
            Circle(radius=0.4).set_fill(color="#808080", opacity=1.0).set_stroke(width=0).shift(RIGHT * 0.4),
            Circle(radius=0.3).set_fill(color="#808080", opacity=1.0).set_stroke(width=0).shift(RIGHT * 0.8)
        )
        self.place_at_grid(cloud_circles, "C6", scale_factor=0.7)
        
        # New 69% fill
        fill_width_69 = 1.5 * 0.69
        new_meter_fill = Rectangle(
            width=fill_width_69, 
            height=0.4, 
            stroke_width=0
        ).set_fill(color=color3, opacity=1.0)
        new_meter_fill.move_to(meter_svg.get_left(), aligned_edge=LEFT).shift(RIGHT * 0.1)
        
        new_rain_label = Text("69% Chance of Rain", font_size=22, color=color3)
        self.place_at_grid(new_rain_label, "C3", scale_factor=0.8)
        
        self.play(FadeIn(cloud_circles))
        self.play(
            Transform(meter_fill, new_meter_fill),
            Transform(rain_label, new_rain_label),
            run_time=2
        )
        self.wait(3)
