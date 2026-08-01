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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "The Riemann Hypothesis: The Critical Line",
            [
                "In this warped landscape, certain points drop to zero.",
                "These 'sinkholes' are the non-trivial zeros of the function.",
                "Riemann predicted they all lie on a single vertical line.",
                "This 'critical line' sits exactly at real part one-half.",
                "Proving this remains mathematics' most famous unsolved mystery."
            ]
        )

        # Colors for highlights
        line_colors = ["#88C0D0", "#A3BE8C", "#EBCB8B", "#D08770", "#B48EAD"]
        critical_line_color = "#FFD700"
        zero_color = "#ECEFF4"
        flash_color = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Zoom into the warped complex plane to reveal 'sinkholes' (zeros).
        plane_color = "#4C566A"
        warped_landscape = VGroup()
        for i in range(-2, 3):
            # Warped horizontal-ish lines
            warped_landscape.add(ParametricFunction(
                lambda t, i=i: np.array([t, 0.1 * np.sin(t * 2) + i * 0.8, 0]),
                t_range=[-2, 2], color=plane_color, stroke_width=1
            ))
            # Warped vertical-ish lines
            warped_landscape.add(ParametricFunction(
                lambda t, i=i: np.array([0.1 * np.cos(t * 2) + i * 0.8, t, 0]),
                t_range=[-2, 2], color=plane_color, stroke_width=1
            ))
        
        # Position warped landscape in the right area
        self.place_in_area(warped_landscape, "B2", "F6")

        self.play(
            self.lecture[0].animate.set_color(line_colors[0]),
            FadeIn(warped_landscape, scale=1.2),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # These 'sinkholes' are the non-trivial zeros of the function.
        # Mark several zero points.
        # Issue 48: Move zeros to 'B4', 'C4', 'D4', 'E4', 'F4'.
        zeros = VGroup(*[Dot(radius=0.1, color=zero_color) for _ in range(5)])
        self.place_at_grid(zeros[0], "B4")
        self.place_at_grid(zeros[1], "C4")
        self.place_at_grid(zeros[2], "D4")
        self.place_at_grid(zeros[3], "E4")
        self.place_at_grid(zeros[4], "F4")

        self.play(
            self.lecture[1].animate.set_color(line_colors[1]),
            AnimationGroup(*[FadeIn(z, scale=0.5) for z in zeros], lag_ratio=0.2),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Riemann predicted they all lie on a single vertical line.
        # Draw a prominent vertical 'critical line' at Re(s) = 0.5.
        # Issue 48: Position critical line in area 'B4' to 'F4'.
        critical_line = Line(UP * 2, DOWN * 2, color=critical_line_color, stroke_width=5)
        self.place_in_area(critical_line, "B4", "F4")

        self.play(
            self.lecture[2].animate.set_color(line_colors[2]),
            Create(critical_line),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # This 'critical line' sits exactly at real part one-half.
        # Label the line 'Re(s) = 1/2' in #FFD700.
        # Issue 48: Move the line label 'Re(s) = 1/2' to 'B5' (scale 0.8).
        line_label = Text("Re(s) = 1/2", color=critical_line_color, font_size=28)
        self.place_at_grid(line_label, "B5", scale_factor=0.8)

        self.play(
            self.lecture[3].animate.set_color(line_colors[3]),
            Write(line_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Proving this remains mathematics' most famous unsolved mystery.
        # Flash the zero points in #FFFFFF.
        self.play(
            self.lecture[4].animate.set_color(line_colors[4]),
            Flash(zeros[2], color=flash_color, line_length=0.3, flash_radius=0.4),
            AnimationGroup(*[z.animate.set_color(flash_color).scale(1.3) for z in zeros], lag_ratio=0.1),
            run_time=2
        )
        # Revert zeros to indicate they are still there
        self.play(
            AnimationGroup(*[z.animate.set_color(zero_color).scale(1/1.3) for z in zeros], lag_ratio=0.05),
            run_time=1
        )
        self.wait(2)
