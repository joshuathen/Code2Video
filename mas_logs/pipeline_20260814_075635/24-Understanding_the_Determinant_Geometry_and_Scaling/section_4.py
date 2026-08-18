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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Zero Determinant: Dimensional Collapse", [
            "Zero determinant collapses space into lower dimensions.",
            "Shapes flatten into lines or single points.",
            "This indicates the transformation is non-invertible."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Draw a 2D grid
        grid = NumberPlane(x_range=[-2, 2], y_range=[-2, 2], background_line_style={"stroke_color": WHITE, "stroke_width": 1})
        self.place_in_area(grid, "D2", "F6", scale_factor=0.4)
        self.play(Create(grid))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Show vectors squashing onto a line
        line = Line(start=[-1, -1, 0], end=[1, 1, 0], color="#FF00FF")
        self.place_in_area(line, "D2", "F6", scale_factor=0.4)
        
        # Animate grid squashing
        self.play(
            grid.animate.apply_matrix([[1, 1], [0, 0]]),
            Create(line),
            run_time=2
        )
        self.lecture[1].set_color("#FF00FF")

        # === Animation for Lecture Line 3 ===
        # Label determinant as zero
        det_label = MathTex(r"\det(A) = 0", color="#FF0000")
        self.place_at_grid(det_label, "B5", scale_factor=0.8)
        self.play(Write(det_label))
        self.lecture[2].set_color("#FF0000")
        self.wait(2)
