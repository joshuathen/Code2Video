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
        # Setup layout
        lecture_lines = [
            "Imagine Leo the Lion walking across a field.",
            "A position graph shows his location over time.",
            "A velocity graph tracks his constant speed."
        ]
        self.setup_layout("Prerequisites: The Language of Motion", lecture_lines)

        # Colors
        LION_COLOR = "#E3CF57"
        POS_COLOR = "#00FF00"
        VEL_COLOR = "#00FFFF"

        # === Animation for Lecture Line 1 ===
        # Highlight Line 1
        self.play(self.lecture[0].animate.set_color(LION_COLOR))
        
        # Lion Icon - Using provided asset
        lion = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/lion.svg")
        lion.set_color(LION_COLOR)
        # Issue 29: Adjust scale to 0.4
        self.place_at_grid(lion, 'D1', scale_factor=0.4)
        
        # Ground line
        ground = Line(self.grid['D1'] + LEFT*0.5, self.grid['D6'] + RIGHT*0.5, color=GREY_C)
        self.add(ground)
        
        time_tracker = ValueTracker(0)
        
        def update_lion(obj):
            t = time_tracker.get_value()
            start_pos = self.grid['D1']
            end_pos = self.grid['D6']
            obj.move_to(start_pos + (end_pos - start_pos) * (t/4))
            
        lion.add_updater(update_lion)
        self.add(lion)
        
        self.play(time_tracker.animate.set_value(4), run_time=3, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight Line 2
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(POS_COLOR)
        )
        
        # Position Axes
        pos_axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 4, 1],
            x_length=3,
            y_length=2,
            axis_config={"include_tip": False, "font_size": 18}
        ).add_coordinates(label_constructor=Text)
        
        pos_label = Text("Position (m)", font_size=16, color=POS_COLOR)
        pos_axes_group = VGroup(pos_axes, pos_label)
        pos_label.next_to(pos_axes, UP, buff=0.1)
        
        # Issue 27: Adjusted area and scale
        self.place_in_area(pos_axes_group, 'B1', 'C6', scale_factor=0.7)
        self.play(Create(pos_axes), Write(pos_label))
        
        # Position Line (Linear growth: y = x)
        pos_line = Line(pos_axes.c2p(0, 0), pos_axes.c2p(0, 0), color=POS_COLOR)
        def update_pos_line(line):
            t = time_tracker.get_value()
            line.set_points_as_corners([pos_axes.c2p(0, 0), pos_axes.c2p(t, t)])
            
        pos_line.add_updater(update_pos_line)
        self.add(pos_line)
        
        # Reset lion and time for visualization
        lion.remove_updater(update_lion)
        time_tracker.set_value(0)
        lion.add_updater(update_lion)
        
        self.play(time_tracker.animate.set_value(4), run_time=3, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight Line 3
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(VEL_COLOR)
        )
        
        # Velocity Axes
        vel_axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 2, 1],
            x_length=3,
            y_length=1.5,
            axis_config={"include_tip": False, "font_size": 18}
        ).add_coordinates(label_constructor=Text)
        
        vel_label = Text("Velocity (m/s)", font_size=16, color=VEL_COLOR)
        vel_axes_group = VGroup(vel_axes, vel_label)
        vel_label.next_to(vel_axes, UP, buff=0.1)
        
        # Issue 28: Adjusted area and scale
        self.place_in_area(vel_axes_group, 'E2', 'F6', scale_factor=0.7)
        self.play(Create(vel_axes), Write(vel_label))
        
        # Velocity Line (Constant growth: y = 1)
        vel_line = Line(vel_axes.c2p(0, 1), vel_axes.c2p(0, 1), color=VEL_COLOR)
        def update_vel_line(line):
            t = time_tracker.get_value()
            line.set_points_as_corners([vel_axes.c2p(0, 1), vel_axes.c2p(t, 1)])
            
        vel_line.add_updater(update_vel_line)
        self.add(vel_line)
        
        # Reset everything for final sweep
        lion.remove_updater(update_lion)
        time_tracker.set_value(0)
        lion.add_updater(update_lion)
        
        # Position and velocity lines update via time_tracker and updaters
        self.play(time_tracker.animate.set_value(4), run_time=4, rate_func=linear)
        self.wait(2)
