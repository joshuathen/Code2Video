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
        # Setup title and lecture lines
        lecture_lines = [
            'The model calculates the probability of the next word.',
            'It functions like a high-stakes "Wheel of Fortune".',
            'Higher probabilities occupy larger slices of the wheel.',
            'Lex spins to pick the most likely next token.',
            'This cycle repeats to build full sentences and paragraphs.'
        ]
        self.setup_layout("Output: Next-Token Prediction", lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        prompt_text = Text("The sky is...", font_size=36, color=WHITE)
        # Position prompt text relative to lecture group on the left
        prompt_text.next_to(self.lecture, DOWN, buff=1.0, aligned_edge=LEFT)
        self.play(Write(prompt_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#0000FF") # Matching blue sector

        # Create the sectors for the Wheel of Fortune
        s1 = Sector(radius=2, angle=0.8*TAU, start_angle=0, color="#0000FF", stroke_width=2, stroke_color=WHITE)
        s2 = Sector(radius=2, angle=0.15*TAU, start_angle=0.8*TAU, color="#888888", stroke_width=2, stroke_color=WHITE)
        s3 = Sector(radius=2, angle=0.05*TAU, start_angle=0.95*TAU, color="#FF0000", stroke_width=2, stroke_color=WHITE)
        
        wheel_group = VGroup(s1, s2, s3)
        # Expansion requested to B1-F6 and scale 0.9
        self.place_in_area(wheel_group, 'B1', 'F6', scale_factor=0.9)
        self.play(Create(wheel_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#0000FF")

        # Create labels inside sectors
        # Helper for polar to cartesian
        def get_sector_pos(angle_prop, radius_prop=0.6):
            angle = angle_prop * TAU
            return np.array([np.cos(angle), np.sin(angle), 0]) * (2 * radius_prop)

        # Adjusted font sizes and radius_prop for better legibility and avoiding borders
        l1 = Text("blue", font_size=32, color=WHITE).move_to(wheel_group.get_center() + get_sector_pos(0.4, 0.5))
        l2 = Text("cloudy", font_size=24, color=WHITE).move_to(wheel_group.get_center() + get_sector_pos(0.875, 0.65))
        l3 = Text("falling", font_size=20, color=WHITE).move_to(wheel_group.get_center() + get_sector_pos(0.975, 0.75))
        
        labels = VGroup(l1, l2, l3)
        self.play(FadeIn(labels))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)

        # Create pointer arrow
        # Adjusted starting height to account for the new wheel area B1-F6
        pointer = Arrow(
            start=wheel_group.get_center() + UP * 2.8, 
            end=wheel_group.get_center() + UP * 1.8, 
            color=WHITE, 
            buff=0,
            stroke_width=8
        )
        self.add(pointer)
        
        # Rotate the pointer over the pie chart, stopping in the blue segment (start_angle 0 to 0.8 TAU)
        # Pointer starts at UP (TAU/4 = 0.25 TAU). 0.25 TAU is already in the blue segment.
        # We will rotate it 2 full turns and a bit more to land back in the blue segment.
        self.play(
            Rotate(pointer, angle=-2 * TAU - 0.1 * TAU, about_point=wheel_group.get_center(), run_time=2.5, rate_func=exponential_decay)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)

        # Update prompt and fade out wheel
        final_prompt = Text("The sky is blue", font_size=36, color=WHITE)
        final_prompt.move_to(prompt_text.get_left(), aligned_edge=LEFT)
        
        self.play(
            Transform(prompt_text, final_prompt),
            FadeOut(wheel_group),
            FadeOut(labels),
            FadeOut(pointer)
        )
        self.wait(2)
