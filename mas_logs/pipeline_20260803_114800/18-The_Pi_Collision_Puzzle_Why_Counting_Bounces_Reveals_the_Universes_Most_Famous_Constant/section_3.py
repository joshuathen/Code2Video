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
        self.setup_layout("Mapping to Phase Space: The Velocity Vector", [
            "We plot velocities on a two-dimensional graph.",
            "Each point represents the current state of both blocks.",
            "Collisions cause this state point to jump around."
        ])
        
        # Issue 40: Anchor the title
        self.place_in_area(self.title, 'A1', 'A6')

        # Colors
        V1_COLOR = "#0000FF"
        V2_COLOR = "#00FF00"
        POINT_COLOR = "#FFFF00"
        TRAIL_COLOR = "#FFFFFF"
        HIGHLIGHT_COLOR = YELLOW

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        # Issue 38: Fix axes positioning
        axes = Axes(
            x_range=[-2.2, 2.2, 1],
            y_range=[-2.2, 2.2, 1],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": True},
            x_axis_config={"color": V1_COLOR},
            y_axis_config={"color": V2_COLOR}
        )
        x_label = MathTex("v_1", color=V1_COLOR).scale(0.8)
        y_label = MathTex("v_2", color=V2_COLOR).scale(0.8)
        
        # Positioning axes
        self.place_in_area(axes, 'B3', 'F6', scale_factor=0.9)
        
        # Position labels near axes tips within 1 unit
        x_label.next_to(axes.x_axis.get_end(), DOWN, buff=0.1)
        y_label.next_to(axes.y_axis.get_end(), LEFT, buff=0.1)
        
        axes_group = VGroup(axes, x_label, y_label)
        self.play(Create(axes_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        # Initial point at (v1_initial, 0). 
        v1_initial = -1.5
        v2_initial = 0
        point = Dot(axes.c2p(v1_initial, v2_initial), color=POINT_COLOR, radius=0.1)
        
        # The trail follows the point and uses #FFFFFF
        trail = TracedPath(point.get_center, stroke_color=TRAIL_COLOR, stroke_width=2)
        
        # Issue 29 & 39: Asset Integration and Coordinate Label placement
        # Label "System State" and Asset icon
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg
        asset_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg")
        state_text = Text("System State", font_size=20, color=POINT_COLOR)
        state_label_container = VGroup(state_text, asset_icon).arrange(RIGHT, buff=0.2)
        
        # Place this fixed part of the label at B4 as requested
        self.place_at_grid(state_label_container, 'B4', scale_factor=0.7)
        
        # The dynamic coordinate label
        v1_val = ValueTracker(v1_initial)
        v2_val = ValueTracker(v2_initial)
        
        # Using DecimalNumber for performance and linking to ValueTracker
        coord_label = VGroup(
            Text("(", font_size=20),
            DecimalNumber(v1_val.get_value(), num_decimal_places=2, color=V1_COLOR),
            Text(", ", font_size=20),
            DecimalNumber(v2_val.get_value(), num_decimal_places=2, color=V2_COLOR),
            Text(")", font_size=20)
        ).arrange(RIGHT, buff=0.05).scale(0.8)
        
        # Position the dynamic label at the grid cell adjacent to the point (Issue 39)
        # Instead of 'B4' (which is used by System State), let's use 'B5'
        self.place_at_grid(coord_label, 'B5', scale_factor=0.7)
        
        # Add updaters for the dynamic values
        def update_v1(obj):
            obj.set_value(v1_val.get_value())
        def update_v2(obj):
            obj.set_value(v2_val.get_value())
        coord_label[1].add_updater(update_v1)
        coord_label[3].add_updater(update_v2)

        self.add(trail)
        self.play(FadeIn(point), FadeIn(state_label_container), FadeIn(coord_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        # Physics logic for animation visualization (m1=10, m2=1)
        m1, m2 = 10.0, 1.0
        
        def get_next_velocities(v1, v2, type="block"):
            if type == "block":
                v1_new = ((m1 - m2) / (m1 + m2)) * v1 + (2 * m2 / (m1 + m2)) * v2
                v2_new = (2 * m1 / (m1 + m2)) * v1 - ((m1 - m2) / (m1 + m2)) * v2
                return v1_new, v2_new
            else:
                return v1, -v2

        # Step-by-step jump animation
        current_v1, current_v2 = v1_initial, v2_initial
        steps = [("block", 1.2), ("wall", 0.6), ("block", 1.2), ("wall", 0.6)]
        
        for step_type, r_time in steps:
            next_v1, next_v2 = get_next_velocities(current_v1, current_v2, step_type)
            self.play(
                point.animate.move_to(axes.c2p(next_v1, next_v2)),
                v1_val.animate.set_value(next_v1),
                v2_val.animate.set_value(next_v2),
                run_time=r_time,
                rate_func=linear
            )
            current_v1, current_v2 = next_v1, next_v2
            self.wait(0.2)

        self.wait(1)
        
        # Cleanup
        coord_label[1].remove_updater(update_v1)
        coord_label[3].remove_updater(update_v2)
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)
