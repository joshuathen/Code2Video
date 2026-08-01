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
        # Setup Layout
        title_text = "The Sequential Challenge: Two Steps, One Move"
        lecture_lines = [
            'Suppose we rotate then shear our character.',
            'Applying two separate matrices takes two steps.',
            'We want a single Master Matrix instead.',
            'This shortcut is called matrix composition.',
            'It skips the middle step entirely.'
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Show Leo [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/leo.svg] at the origin on a standard grid.
        
        coordinate_grid = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=4,
            y_length=4,
            background_line_style={"stroke_color": BLUE_D, "stroke_opacity": 0.6}
        )
        # [Issue 34, 36] Fixing layout and grid utilization
        self.place_in_area(coordinate_grid, 'B2', 'F5', scale_factor=0.9)
        
        # Align origin with D4 to satisfy Issue 35's central anchor while maintaining "at origin" logic
        grid_origin = coordinate_grid.c2p(0, 0)
        
        leo = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/leo.svg")
        # [Issue 35] Scale factor 0.7 and central anchor logic
        leo.move_to(grid_origin).scale(0.7)
        
        self.play(
            Create(coordinate_grid),
            FadeIn(leo),
            self.lecture[0].animate.set_color(WHITE),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Applying two separate matrices takes two steps.
        # Rotate Leo [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/leo.svg] 90 degrees CCW (Transformation A) in #00FF00.
        # Shear the rotated Leo [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/leo.svg] horizontally (Transformation B) in #00FFFF.
        
        rot_mat = [[0, -1], [1, 0]]
        shear_mat = [[1, 1], [0, 1]]
        
        self.play(
            self.lecture[1].animate.set_color(YELLOW),
            leo.animate.apply_matrix(rot_mat, about_point=grid_origin).set_color("#00FF00"),
            run_time=1.5
        )
        self.wait(0.5)
        
        self.play(
            leo.animate.apply_matrix(shear_mat, about_point=grid_origin).set_color("#00FFFF"),
            run_time=1.5
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # We want a single Master Matrix instead.
        # Clear the screen (visual area) and display the text 'Master Matrix C' in #FFFFFF.
        
        master_label = Text("Master Matrix C", color=WHITE, font_size=32)
        self.place_in_area(master_label, 'C3', 'D4', scale_factor=1.2)
        
        self.play(
            FadeOut(leo),
            FadeOut(coordinate_grid),
            FadeIn(master_label),
            self.lecture[2].animate.set_color(WHITE),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # This shortcut is called matrix composition.
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            Indicate(master_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # It skips the middle step entirely.
        # Apply Transformation C to the original Leo [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/leo.svg], 
        # moving him directly to the final sheared-rotated state.
        
        # Bring back the original context
        plane_final = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=4,
            y_length=4,
            background_line_style={"stroke_color": BLUE_D, "stroke_opacity": 0.6}
        )
        self.place_in_area(plane_final, 'B2', 'F5', scale_factor=0.9)
        grid_origin_final = plane_final.c2p(0, 0)
        
        leo_final = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/leo.svg")
        leo_final.move_to(grid_origin_final).scale(0.7)
        
        # Combined matrix C = Shear * Rotation
        comp_mat = [[1, -1], [1, 0]]
        
        self.play(
            FadeOut(master_label),
            FadeIn(plane_final),
            FadeIn(leo_final),
            self.lecture[4].animate.set_color(BLUE),
            run_time=1.0
        )
        self.wait(0.5)
        
        self.play(
            leo_final.animate.apply_matrix(comp_mat, about_point=grid_origin_final).set_color(WHITE),
            run_time=2
        )
        self.wait(2)
