from manim import *

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
        # Setup title and lecture
        self.setup_layout(
            "The Dimensional Ladder (Prerequisites)", 
            [
                'A point is a zero-dimensional location.', 
                'Dragging a point creates a one-dimensional line.', 
                'Shifting the line creates a two-dimensional square.', 
                'Stacking squares builds a three-dimensional cube.', 
                'Each new dimension adds a new direction of movement.'
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.lecture[0].set_color(YELLOW)
        
        # A single white point (#FFFFFF) appears in the center of the screen.
        point = Dot(color="#FFFFFF", radius=0.08)
        self.place_in_area(point, "B2", "E5")
        self.play(FadeIn(point))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second line
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # The point stretches horizontally to become a 1D white line (#FFFFFF).
        line_1d = Line(LEFT, RIGHT, color="#FFFFFF", stroke_width=4)
        # Issue 33: Fix scale inconsistency (from 1.5 to 1.0)
        self.place_in_area(line_1d, "B2", "E5", scale_factor=1.0)
        
        self.play(ReplacementTransform(point, line_1d))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third line
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # The line stretches vertically to form a 2D white square (#FFFFFF).
        square_2d = Square(side_length=2, color="#FFFFFF", fill_opacity=0.2)
        # Issue 33: Fix scale inconsistency (from 1.2 to 1.0)
        self.place_in_area(square_2d, "B2", "E5", scale_factor=1.0)
        
        self.play(ReplacementTransform(line_1d, square_2d))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight fourth line
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Create a second white square shifted diagonally and connect its corners
        # to the first square to form a 3D cube wireframe (#FFFFFF).
        # We manually build a projection of a cube.
        sq1 = Square(side_length=2, color="#FFFFFF").set_stroke(width=2)
        sq2 = Square(side_length=2, color="#FFFFFF").set_stroke(width=2)
        
        # Shift sq2
        sq2.shift(UP*0.5 + RIGHT*0.5)
        
        cube_group = VGroup(sq1, sq2)
        # Add corners
        for p1, p2 in zip(sq1.get_vertices(), sq2.get_vertices()):
            cube_group.add(Line(p1, p2, color="#FFFFFF", stroke_width=2))
        
        self.place_in_area(cube_group, "B2", "E5", scale_factor=1.0)
        
        self.play(ReplacementTransform(square_2d, cube_group))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Highlight fifth line
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # A red dot (#FF0000) on a 1D line stops at a barrier; 
        # it then moves onto a 2D plane and slides around the barrier.
        self.play(FadeOut(cube_group))
        
        # 1D Setup
        line_1d_path = Line(LEFT*2, RIGHT*2, color=WHITE)
        # Issue 34: Fix spatial balance by shifting area (from B2-B5 to B3-B6)
        self.place_in_area(line_1d_path, "B3", "B6", scale_factor=1.0)
        barrier_1d = Line(UP*0.2, DOWN*0.2, color=BLUE).move_to(line_1d_path.get_center())
        red_dot = Dot(color="#FF0000")
        red_dot.move_to(line_1d_path.get_left())
        
        self.play(Create(line_1d_path), Create(barrier_1d))
        self.play(FadeIn(red_dot))
        
        # Dot stops at barrier
        self.play(red_dot.animate.move_to(barrier_1d.get_center() + LEFT*0.1), run_time=1)
        self.wait(0.5)
        
        # Transition to 2D
        plane_2d = Square(side_length=3, color=WHITE, fill_opacity=0.1)
        # Issue 35: Fix crowding by shifting area (from C2-F5 to C3-F6)
        self.place_in_area(plane_2d, "C3", "F6", scale_factor=1.0)
        barrier_2d = Line(UP*1, DOWN*1, color=BLUE).move_to(plane_2d.get_center())
        
        self.play(
            FadeOut(line_1d_path), 
            FadeOut(barrier_1d),
            FadeIn(plane_2d),
            FadeIn(barrier_2d),
            red_dot.animate.move_to(plane_2d.get_left() + RIGHT*0.2)
        )
        
        # Slides around the barrier
        path = VMobject()
        start = red_dot.get_center()
        mid_top = barrier_2d.get_top() + UP*0.3
        end = plane_2d.get_right() - RIGHT*0.2
        path.set_points_as_corners([start, mid_top, end])
        
        self.play(MoveAlongPath(red_dot, path), run_time=2)
        self.wait(2)
