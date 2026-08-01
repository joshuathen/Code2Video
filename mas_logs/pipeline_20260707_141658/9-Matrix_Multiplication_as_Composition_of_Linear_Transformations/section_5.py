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
        # Section data
        title_text = "Order Matters: Non-Commutativity"
        lecture_lines = [
            "In matrix multiplication, the order of operations matters.",
            "Rotating then stretching differs from stretching then rotating.",
            "Matrix AB is usually not equal to BA.",
            "Geometry shows why multiplication is not commutative."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Matrix Definitions
        # A: Rotation 90 degrees Counter-Clockwise
        # B: Scaling along X-axis by a factor of 2
        rot_mat = [[0, -1], [1, 0]]
        stretch_mat = [[2, 0], [0, 1]]

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Setup two planes side-by-side for comparison
        plane_ba = NumberPlane(
            x_range=[-2.5, 2.5, 1], y_range=[-2.5, 2.5, 1], 
            x_length=2.4, y_length=2.4,
            background_line_style={"stroke_opacity": 0.4}
        ).set_color(GRAY)
        # Fix: Adjusted scale_factor to 0.75 to prevent clipping (Issue 33)
        self.place_in_area(plane_ba, 'A1', 'C3', scale_factor=0.75)
        
        label_ba = Text("BA: Rotate then Stretch", font_size=16, color=YELLOW)
        label_ba.next_to(plane_ba, UP, buff=0.1)

        plane_ab = NumberPlane(
            x_range=[-2.5, 2.5, 1], y_range=[-2.5, 2.5, 1], 
            x_length=2.4, y_length=2.4,
            background_line_style={"stroke_opacity": 0.4}
        ).set_color(GRAY)
        # Fix: Adjusted scale_factor to 0.75 to prevent clipping (Issue 33)
        self.place_in_area(plane_ab, 'A4', 'C6', scale_factor=0.75)
        
        label_ab = Text("AB: Stretch then Rotate", font_size=16, color=GREEN)
        label_ab.next_to(plane_ab, UP, buff=0.1)

        self.play(
            Create(plane_ba), Create(plane_ab),
            Write(label_ba), Write(label_ab),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/cat.png
        def create_cat():
            return ImageMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/cat.png").scale(0.25)

        cat_ba = create_cat()
        cat_ba.move_to(plane_ba.c2p(1, 0)) # Start at point (1, 0)
        
        cat_ab = create_cat()
        cat_ab.move_to(plane_ab.c2p(1, 0)) # Start at point (1, 0)

        self.play(FadeIn(cat_ba), FadeIn(cat_ab))
        self.wait(1)

        # Scenario BA: A (Rotate) then B (Stretch)
        # Apply Rotation (Matrix A)
        self.play(cat_ba.animate.apply_matrix(rot_mat, about_point=plane_ba.c2p(0, 0)), run_time=1.5)
        self.wait(0.5)
        # Apply Stretch (Matrix B)
        self.play(cat_ba.animate.apply_matrix(stretch_mat, about_point=plane_ba.c2p(0, 0)), run_time=1.5)
        
        # Highlight Scenario 1 Result (Issue 22)
        highlight_ba = Circle(radius=0.3, color="#FFD700", fill_opacity=0.3, stroke_width=2).move_to(cat_ba.get_center())
        self.play(Create(highlight_ba))

        # Scenario AB: B (Stretch) then A (Rotate)
        # Apply Stretch (Matrix B)
        self.play(cat_ab.animate.apply_matrix(stretch_mat, about_point=plane_ab.c2p(0, 0)), run_time=1.5)
        self.wait(0.5)
        # Apply Rotation (Matrix A)
        self.play(cat_ab.animate.apply_matrix(rot_mat, about_point=plane_ab.c2p(0, 0)), run_time=1.5)
        
        # Highlight Scenario 2 Result (Issue 22)
        highlight_ab = Circle(radius=0.3, color="#ADFF2F", fill_opacity=0.3, stroke_width=2).move_to(cat_ab.get_center())
        self.play(Create(highlight_ab))

        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(RED)

        # Resulting coordinates
        # Fix: Adjusted scale_factor to 0.8 for coordinate labels (Issue 35)
        res_ba_label = Text("(0, 1)", font_size=24, color="#FFD700")
        self.place_at_grid(res_ba_label, 'D2', scale_factor=0.8)
        
        res_ab_label = Text("(0, 2)", font_size=24, color="#ADFF2F")
        self.place_at_grid(res_ab_label, 'D5', scale_factor=0.8)

        # Display final Inequality
        inequality_text = Text("AB ≠ BA", font_size=42, color=RED)
        # Fix: Adjusted area and scale_factor for inequality (Issue 34)
        self.place_in_area(inequality_text, 'E2', 'F5', scale_factor=0.8)

        self.play(
            Write(res_ba_label),
            Write(res_ab_label)
        )
        self.play(Write(inequality_text))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)

        # Emphasize the geometric difference
        self.play(Indicate(inequality_text), Flash(inequality_text, color=RED, line_length=0.3))
        # Highlight both cats (Issue 22)
        self.play(
            Circumscribe(cat_ba, color="#FFD700"),
            Circumscribe(cat_ab, color="#ADFF2F")
        )
        self.wait(2)
