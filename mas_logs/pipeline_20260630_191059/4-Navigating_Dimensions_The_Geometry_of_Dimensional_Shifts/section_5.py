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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup title and lecture lines
        title_text = "Application: The Dimensional Keyhole"
        lecture_lines = [
            "Navigate restricted spaces by shifting an object's orientation.",
            "Align the thin axis with the narrow opening.",
            "Dimensional perspective reveals paths through seemingly impossible barriers."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        barrier_color = "#E0E0E0"  # Light Grey
        t_shape_color = "#FFA500"  # Orange
        highlight_color = YELLOW

        # Create Visual Elements
        # Assets
        keyhole_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/keyhole.svg"
        keyhole = SVGMobject(keyhole_path).set_color(barrier_color)
        # Place keyhole at the gap center (D5)
        self.place_at_grid(keyhole, "D5", scale_factor=0.7)

        # Barrier Walls (Ref: Issues 33, 34)
        top_wall = Rectangle(width=0.4, height=3.0, color=barrier_color, fill_opacity=0.6, stroke_width=0)
        bottom_wall = Rectangle(width=0.4, height=2.0, color=barrier_color, fill_opacity=0.6, stroke_width=0)
        
        # Issue 33: Move top wall to Column 5 (A5-C5)
        self.place_in_area(top_wall, "A5", "C5")
        # Issue 34: Move bottom wall to Column 5 (E5-F5)
        self.place_in_area(bottom_wall, "E5", "F5")
        
        # T-shape Artifact
        # Designed to be taller than the gap (gap height ~1.0 unit)
        main_bar = Rectangle(width=0.3, height=1.5, color=t_shape_color, fill_opacity=0.9, stroke_width=0)
        top_bar = Rectangle(width=0.8, height=0.3, color=t_shape_color, fill_opacity=0.9, stroke_width=0).next_to(main_bar, UP, buff=0)
        t_shape = VGroup(main_bar, top_bar).center()
        
        # Issue 32: Initial position D2, scale 0.7
        # At scale 0.7, height = (1.5+0.3)*0.7 = 1.26 (Blocked)
        self.place_at_grid(t_shape, "D2", scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        # Show a keyhole and an orange T-shape trying to pass through but blocked.
        self.lecture[0].set_color(highlight_color)
        self.add(top_wall, bottom_wall, keyhole)
        self.play(FadeIn(t_shape))
        self.wait(0.5)
        
        # Movement toward the wall at Column 5
        # Stop just before collision
        target_collision = self.grid["D5"] + LEFT * 0.6
        self.play(t_shape.animate.move_to(target_collision), run_time=1.5, rate_func=rush_into)
        # Blocked: minor recoil to emphasize collision
        self.play(t_shape.animate.shift(LEFT * 0.3), run_time=0.4, rate_func=wiggle)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Rotate the orange T-shape so its profile matches the keyhole width.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(highlight_color)
        
        # Rotate 90 degrees (now height is 0.8 * 0.7 = 0.56, which fits in 1.0 gap)
        self.play(
            t_shape.animate.rotate(-PI/2),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Slide the rotated T-shape smoothly through the keyhole.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(highlight_color)
        
        # Pass through the keyhole at D5 to D6
        self.play(t_shape.animate.move_to(self.grid["D6"]), run_time=2)
        self.wait(2)

        # Reset highlight
        self.lecture[2].set_color(WHITE)
        self.wait(1)
