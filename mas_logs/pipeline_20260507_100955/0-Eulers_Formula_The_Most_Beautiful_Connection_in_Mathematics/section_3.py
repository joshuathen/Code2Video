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
        # Initialization
        title_text = "Prerequisite 2: Redefining Growth with 'e'"
        lecture_lines = [
            'The number e drives continuous exponential growth.',
            'But i adds a twist to this expansion.',
            'It steers growth sideways into the complex plane.'
        ]
        self.setup_layout(title_text, lecture_lines)
        
        blue_color = "#00BFFF"
        yellow_color = "#FFFF00"
        
        # Reference coordinates
        origin_pos = self.grid["D2"]
        
        # Visual support: Axes for context
        h_axis = Line(self.grid["D1"], self.grid["D6"], color=GRAY_D, stroke_width=2)
        v_axis = Line(self.grid["F2"], self.grid["A2"], color=GRAY_D, stroke_width=2)
        
        re_label = Text("Re", font_size=18, color=GRAY_D).next_to(self.grid["D6"], DOWN, buff=0.1)
        im_label = Text("Im", font_size=18, color=GRAY_D).next_to(self.grid["A2"], LEFT, buff=0.1)
        self.add(h_axis, v_axis, re_label, im_label)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(blue_color)
        
        # Create growing vector representing growth e^x along the real axis
        # Growing from D2 (origin) to D5
        growth_vector = Arrow(start=origin_pos, end=self.grid["D5"], color=blue_color, buff=0, stroke_width=4)
        vector_label = Text("e^x", color=blue_color, font_size=24)
        self.place_at_grid(vector_label, "C4", scale_factor=1.0)
        
        self.play(GrowArrow(growth_vector), FadeIn(vector_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(yellow_color)
        
        # Show rotation symbol (arc) labeled 'i' appearing at the tip of the vector
        rotation_arc = Arc(radius=0.4, start_angle=0, angle=PI/2, color=yellow_color)
        rotation_arc.move_to(self.grid["D5"])
        i_label = Text("i", color=yellow_color, font_size=24)
        self.place_at_grid(i_label, "C6", scale_factor=1.0)
        
        # Update label to e^ix and move to B1 to avoid overlap with Im axis (Issue 31/43)
        new_vector_label = Text("e^ix", color=yellow_color, font_size=24)
        self.place_at_grid(new_vector_label, "B1", scale_factor=1.2)
        
        self.play(
            Create(rotation_arc), 
            FadeIn(i_label),
            Transform(vector_label, new_vector_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(yellow_color)
        
        # Shift the direction of the blue growth vector by 90 degrees to point upwards
        # New target is B2 (vertical growth)
        upward_vector = Arrow(start=origin_pos, end=self.grid["B2"], color=blue_color, buff=0, stroke_width=4)
        
        self.play(
            Transform(growth_vector, upward_vector),
            FadeOut(rotation_arc),
            FadeOut(i_label),
            run_time=1.5
        )
        self.wait(2)
