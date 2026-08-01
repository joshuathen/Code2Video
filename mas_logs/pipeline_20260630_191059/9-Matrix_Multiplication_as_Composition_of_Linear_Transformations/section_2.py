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
        # Setup the layout with lecture lines from the storyboard
        self.setup_layout(
            "The Sequential Challenge: Two Steps in a Row",
            [
                "- First, we rotate Momo 90 degrees counter-clockwise.",
                "- Next, we apply a horizontal shear to him.",
                "- Can we find one matrix for both steps?"
            ]
        )

        # Colors for consistency and lecture line matching
        ROT_COLOR = "#58C4DD"    # Light blue for Rotation
        SHEAR_COLOR = "#83C167"  # Light green for Shear
        MOMO_COLOR = "#FFD700"   # Gold for Momo
        MASTER_COLOR = "#FF8C00" # Orange for the Master Matrix question
        
        # Create Momo using shapes (persistent mobjects)
        momo_body = Square(side_length=1, fill_opacity=0.8, color=MOMO_COLOR)
        eye_l = Dot(point=[-0.2, 0.2, 0], color=BLACK).scale(0.5)
        eye_r = Dot(point=[0.2, 0.2, 0], color=BLACK).scale(0.5)
        smile = Arc(radius=0.2, start_angle=-TAU/4 - 0.5, angle=1, color=BLACK)
        momo = VGroup(momo_body, eye_l, eye_r, smile)
        
        # Create Coordinate Plane to visualize transformations
        plane = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=4,
            y_length=4,
            background_line_style={
                "stroke_color": GREY,
                "stroke_width": 2,
                "stroke_opacity": 0.5
            }
        )
        
        # Group Momo and Plane so they transform together
        transformation_group = VGroup(plane, momo)
        
        # Initial placement using the 6x6 grid system
        # Fix for Issue 31: Reduced scale factor to 0.65 to avoid cramped layout
        self.place_in_area(transformation_group, 'A1', 'F6', scale_factor=0.65)
        
        # Store the center point for the transformations
        group_center = transformation_group.get_center().copy()

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line and perform rotation
        self.lecture[0].set_color(ROT_COLOR)
        self.add(transformation_group)
        
        # 90 degrees CCW rotation matrix: [[0, -1], [1, 0]]
        rot_matrix = [[0, -1], [1, 0]]
        
        self.play(
            transformation_group.animate.apply_matrix(rot_matrix, about_point=group_center),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition highlight to second line and perform shear
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(SHEAR_COLOR)
        
        # Horizontal shear matrix (k=1): [[1, 1], [0, 1]]
        shear_matrix = [[1, 1], [0, 1]]
        
        self.play(
            transformation_group.animate.apply_matrix(shear_matrix, about_point=group_center),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition highlight to third line for the summary question
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(MASTER_COLOR)
        
        # Pulse animation for emphasis on the final "Sequential Challenge" result
        self.play(
            transformation_group.animate.scale(1.15),
            rate_func=there_and_back,
            run_time=1.5
        )
        self.wait(2)
