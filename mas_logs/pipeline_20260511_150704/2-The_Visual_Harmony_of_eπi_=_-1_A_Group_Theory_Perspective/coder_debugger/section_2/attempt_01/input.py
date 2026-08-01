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
        # Setup title and lecture lines
        title_text = "The 'i' Operator: Rotation, Not Just a Number"
        lecture_lines = [
            'A single rotation by i turns us ninety degrees.',
            'This action places us at the imaginary unit i.',
            'Repeating the rotation lands us on negative one.'
        ]
        self.setup_layout(title_text, lecture_lines)

        # --- Visual Elements Setup ---
        # Origin at D4, Units 2 grid squares wide/high
        # Radius for rotation is 2 units.
        origin_pt = self.grid["D4"]
        pos_1 = self.grid["D6"]
        pos_i = self.grid["B4"]
        pos_neg_1 = self.grid["D2"]

        # Issue 42: Axes scaling and area. 
        # Area A2-F6 is 4 units wide (2 to 6) and 5 units high (A to F).
        # Center of A2-F6 is x=3.5, y=-0.3. 
        # Grid D4 (origin) is x=3.5, y=-0.8.
        # To align origin with D4, we shift the y_range so origin (0) is 0.5 units below center.
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[-2, 3, 1],
            x_length=4.0,
            y_length=5.0,
            axis_config={"color": GRAY_C, "include_tip": True}
        )
        self.place_in_area(axes, 'A2', 'F6')

        # Vector and Labels
        vector = Arrow(origin_pt, pos_1, buff=0, color="#FFFF00", stroke_width=6)
        
        # Point and Labels for rotation milestones
        label_1 = Text("1", font_size=24, color="#FFFFFF")
        self.place_at_grid(label_1, "E6", scale_factor=0.6)
        
        # Issue 43: Precise grid anchor for point 'i'
        point_i = Dot(color="#00FF00")
        self.place_at_grid(point_i, "B4", scale_factor=0.7)
        label_i = Text("i", font_size=24, color="#00FF00")
        self.place_at_grid(label_i, "B5", scale_factor=0.6)
        
        # Issue 44: Precise grid anchor for point '-1'
        point_neg_1 = Dot(color="#FF0000")
        self.place_at_grid(point_neg_1, "D2", scale_factor=0.7)
        label_neg_1 = Text("-1", font_size=24, color="#FF0000")
        self.place_at_grid(label_neg_1, "E2", scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Create(axes), run_time=1)
        self.play(GrowArrow(vector), Write(label_1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Arc to visualize the 90-degree CCW rotation
        arc1 = Arc(radius=2.0, start_angle=0, angle=PI/2, arc_center=origin_pt, color="#00FF00")
        
        self.play(
            Rotate(vector, angle=PI/2, about_point=origin_pt, rate_func=smooth),
            Create(arc1),
            run_time=2
        )
        self.play(
            Flash(point_i, color="#00FF00", flash_radius=0.4),
            FadeIn(label_i)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Second Arc for rotation to -1
        arc2 = Arc(radius=2.0, start_angle=PI/2, angle=PI/2, arc_center=origin_pt, color="#FF0000")
        
        self.play(
            Rotate(vector, angle=PI/2, about_point=origin_pt, rate_func=smooth),
            Create(arc2),
            run_time=2
        )
        self.play(
            Flash(point_neg_1, color="#FF0000", flash_radius=0.4),
            FadeIn(label_neg_1)
        )
        self.wait(2)

        # Reset highlight
        self.lecture[2].set_color(WHITE)
        self.wait(1)
