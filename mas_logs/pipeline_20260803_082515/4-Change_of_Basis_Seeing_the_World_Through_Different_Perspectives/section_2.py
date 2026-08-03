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

class Section2Scene(TeachingScene):
    def construct(self):
        # Setup the title and lecture lines
        self.setup_layout("Prerequisite Check: The Standard Basis", [
            "- We usually describe space using standard unit vectors.",
            "- The vector i and j form our standard grid.",
            "- Every point is a combination of these base steps."
        ])

        # === Animation for Lecture Line 1 ===
        # Show a standard white square grid
        grid = NumberPlane(
            x_range=[-1, 5, 1],
            y_range=[-1, 4, 1],
            background_line_style={
                "stroke_color": WHITE,
                "stroke_width": 2,
                "stroke_opacity": 0.3
            },
            axis_config={"stroke_color": WHITE}
        )
        self.place_in_area(grid, "A1", "F6", scale_factor=0.8)
        
        # Initially color the first line
        self.play(
            Create(grid),
            self.lecture[0].animate.set_color(YELLOW),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate the unit vectors i (1,0) in #FF5555 and j (0,1) in #55FF55
        origin = grid.coords_to_point(0, 0)
        i_end = grid.coords_to_point(1, 0)
        j_end = grid.coords_to_point(0, 1)

        i_vec = Arrow(origin, i_end, buff=0, color="#FF5555", stroke_width=4)
        j_vec = Arrow(origin, j_end, buff=0, color="#55FF55", stroke_width=4)

        # Fix Issue 36: j_hat_label too small and close to vertical axis
        j_hat_label = MathTex(r"\hat{j}", color="#55FF55")
        self.place_at_grid(j_hat_label, 'C1', scale_factor=0.7)
        
        # Fix Issue 37: i_hat_label small and cramped under horizontal axis
        i_hat_label = MathTex(r"\hat{i}", color="#FF5555")
        self.place_at_grid(i_hat_label, 'E3', scale_factor=0.7)

        self.play(
            GrowArrow(i_vec),
            GrowArrow(j_vec),
            Write(i_hat_label),
            Write(j_hat_label),
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Draw the vector [3, 2] by moving 3 units right then 2 units up
        target_point = grid.coords_to_point(3, 2)
        intermediate_point = grid.coords_to_point(3, 0)

        # Component lines to show the "steps"
        step_h = Line(origin, intermediate_point, color="#FF5555", stroke_width=4)
        step_v = Line(intermediate_point, target_point, color="#55FF55", stroke_width=4)
        
        v_vec = Arrow(origin, target_point, buff=0, color="#FFFF55", stroke_width=6)
        
        # Fix Issue 38: coord_label too high/far right
        coord_label = MathTex(r"\begin{bmatrix} 3 \\ 2 \end{bmatrix}", color="#FFFF55")
        self.place_at_grid(coord_label, 'B6', scale_factor=0.7)

        self.play(
            Create(step_h),
            run_time=1
        )
        self.play(
            Create(step_v),
            run_time=1
        )
        self.play(
            GrowArrow(v_vec),
            Write(coord_label),
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFF55"),
            run_time=2
        )
        self.wait(3)
