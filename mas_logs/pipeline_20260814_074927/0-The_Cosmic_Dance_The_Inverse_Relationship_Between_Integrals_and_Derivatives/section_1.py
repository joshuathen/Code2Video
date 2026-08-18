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
        # === Setup ===
        lecture_lines = [
            "- Meet Dash the Cheetah, sprinting across the savannah.",
            "- His position over time creates a distinct curve.",
            "- The slope at any point reveals his instantaneous speed."
        ]
        self.setup_layout("The Prerequisites: Motion and Change", lecture_lines)
        # Apply requested neon cyan color to title
        self.title.set_color("#00FFFF")

        # === Animation for Lecture Line 1 ===
        # Line: "Meet Dash the Cheetah, sprinting across the savannah."
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        
        # Savannah path visualization
        savannah_line = Line(self.grid["F1"] + LEFT*0.5, self.grid["F6"] + RIGHT*0.5, color=GREY_E)
        self.add(savannah_line)
        
        # Dash - Gold circle
        dash = Circle(radius=0.2, color="#FFD700", fill_opacity=0.9)
        self.place_at_grid(dash, "E1")
        
        dash_label = Text("Dash", font_size=18, color="#FFD700")
        dash_label.next_to(dash, UP, buff=0.1)
        
        self.play(FadeIn(dash), FadeIn(dash_label))
        # Sprinting across the savannah
        self.play(
            dash.animate.move_to(self.grid["E6"]),
            dash_label.animate.move_to(self.grid["E6"] + UP*0.3),
            run_time=3,
            rate_func=slow_into
        )
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Line: "His position over time creates a distinct curve."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFD700")
        )
        
        # Position Graph
        pos_axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 5, 1],
            x_length=2.5,
            y_length=2.0,
            axis_config={"include_tip": True, "font_size": 14, "color": WHITE}
        )
        pos_title = Text("Position", font_size=20, color="#FFD700")
        pos_plot = pos_axes.plot(lambda x: 0.15 * x**2, x_range=[0, 5], color="#FFD700")
        
        pos_group = VGroup(pos_axes, pos_title, pos_plot)
        # Position in top-left quadrant of the grid
        self.place_in_area(pos_group, "B1", "C3", scale_factor=0.9)
        pos_title.next_to(pos_axes, UP, buff=0.2)
        
        self.play(FadeIn(pos_axes), FadeIn(pos_title))
        self.play(Create(pos_plot))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line: "The slope at any point reveals his instantaneous speed."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#ADFF2F")
        )
        
        # Speed Graph
        speed_axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 5, 1],
            x_length=2.5,
            y_length=2.0,
            axis_config={"include_tip": True, "font_size": 14, "color": WHITE}
        )
        speed_title = Text("Speed", font_size=20, color="#ADFF2F")
        # Derivative of 0.15x^2 is 0.3x
        speed_plot = speed_axes.plot(lambda x: 0.3 * x, x_range=[0, 5], color="#ADFF2F")
        
        speed_group = VGroup(speed_axes, speed_title, speed_plot)
        # Position in top-right quadrant of the grid
        self.place_in_area(speed_group, "B4", "C6", scale_factor=0.9)
        speed_title.next_to(speed_axes, UP, buff=0.2)
        
        # Connecting arrows between graphs
        # Forward: Differentiate
        diff_arrow = Arrow(
            start=self.grid["C3"] + RIGHT*0.2,
            end=self.grid["C4"] + LEFT*0.2,
            color="#ADFF2F",
            stroke_width=5
        )
        diff_tag = Text("Differentiate", font_size=16, color="#ADFF2F")
        diff_tag.next_to(diff_arrow, UP, buff=0.1)
        
        # Backward: Integrate
        int_arrow = CurvedArrow(
            start_point=self.grid["D4"] + LEFT*0.2,
            end_point=self.grid["D3"] + RIGHT*0.2,
            angle=-TAU/4,
            color="#FF69B4"
        )
        int_tag = Text("Integrate", font_size=16, color="#FF69B4")
        int_tag.next_to(int_arrow, DOWN, buff=0.1)
        
        self.play(FadeIn(speed_axes), FadeIn(speed_title))
        self.play(Create(speed_plot))
        self.play(GrowArrow(diff_arrow), Write(diff_tag))
        self.play(Create(int_arrow), Write(int_tag))
        self.wait(2)
