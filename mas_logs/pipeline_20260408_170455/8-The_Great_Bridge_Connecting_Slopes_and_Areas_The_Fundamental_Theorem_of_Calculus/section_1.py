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
        # Title and Lecture lines setup
        title_text = "The Runner's Puzzle: A Tale of Two Tasks"
        lecture_lines = [
            "Meet Zippy, our fast cheetah runner.",
            "We can track his position or his speed.",
            "How are distance and speed actually connected?"
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Color match: #FFD700
        self.lecture[0].set_color("#FFD700")
        
        # Create Zippy using asset
        zippy_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/cheetah.svg")
        zippy_icon.set_color("#FFD700")
        zippy_label = Text("Zippy", font_size=18, color=WHITE)
        zippy_label.next_to(zippy_icon, UP, buff=0.1)
        zippy = VGroup(zippy_icon, zippy_label)
        
        # Place at grid start (A1) with requested scale (Issue 34, 52)
        self.place_at_grid(zippy, "A1", scale_factor=0.5)
        
        self.play(FadeIn(zippy))
        # Animate across screen using grid (A1 to A6)
        self.play(zippy.animate.move_to(self.grid["A6"]), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Color match: #00FF00 (Position)
        self.lecture[1].set_color("#00FF00")
        
        # Define axes for position and velocity
        # Top half: Position s(t) = 0.5*t^2
        pos_axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 8, 2],
            axis_config={"include_tip": True, "font_size": 16},
            x_length=3.5,
            y_length=2.0
        )
        pos_label = Text("Position s(t)", font_size=14, color="#00FF00")
        pos_label.next_to(pos_axes, UP, buff=0.1)
        pos_graph = pos_axes.plot(lambda t: 0.5 * t**2, x_range=[0, 4], color="#00FF00")
        pos_system = VGroup(pos_axes, pos_graph, pos_label)
        
        # Bottom half: Velocity v(t) = t
        vel_axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 4, 1],
            axis_config={"include_tip": True, "font_size": 16},
            x_length=3.5,
            y_length=2.0
        )
        vel_label = Text("Velocity v(t)", font_size=14, color="#FF4500")
        vel_label.next_to(vel_axes, UP, buff=0.1)
        vel_graph = vel_axes.plot(lambda t: t, x_range=[0, 4], color="#FF4500")
        vel_system = VGroup(vel_axes, vel_graph, vel_label)
        
        # Position the graph systems with requested scales (Issue 35, 36, 52)
        self.place_in_area(pos_system, "A1", "C6", scale_factor=0.8)
        self.place_in_area(vel_system, "D1", "F6", scale_factor=0.8)
        
        self.play(
            FadeOut(zippy),
            Create(pos_axes),
            Create(pos_graph),
            Write(pos_label),
            run_time=2
        )
        self.play(
            Create(vel_axes),
            Create(vel_graph),
            Write(vel_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Color match: #FF4500 (Velocity/Connection)
        self.lecture[2].set_color("#FF4500")
        
        # Visualizing connection: Highlight graphs
        self.play(
            pos_graph.animate.set_stroke(width=8),
            vel_graph.animate.set_stroke(width=8),
            run_time=1
        )
        self.play(
            pos_graph.animate.set_stroke(width=4),
            vel_graph.animate.set_stroke(width=4),
            run_time=1
        )
        
        self.wait(2)
