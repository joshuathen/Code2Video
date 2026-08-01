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
        # Setup the title and lecture lines
        title_text = "The Rule of Rows and Columns"
        lecture_lines = [
            "- Matrix dimensions m-by-n describe the transformation's path.",
            "- The column count n is the starting space dimension.",
            "- The row count m is the target space dimension."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors defined in storyboard
        COLOR_N = "#0000FF"  # Blue (columns, input)
        COLOR_M = "#FF0000"  # Red (rows, output)

        # === Animation for Lecture Line 1 ===
        # Write the dimensions 'm x n'
        dim_m = MathTex("m", font_size=60, color=COLOR_M)
        dim_x = MathTex(r"\times", font_size=40, color=WHITE)
        dim_n = MathTex("n", font_size=60, color=COLOR_N)
        dimensions = VGroup(dim_m, dim_x, dim_n).arrange(RIGHT, buff=0.2)
        
        # Fix for Issue 30: Shift dimensions further right to avoid clutter
        self.place_in_area(dimensions, "A4", "A6", scale_factor=0.8)
        
        self.lecture[0].set_color(YELLOW)
        self.play(Write(dimensions))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight 'n' and start building the portal inputs
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_N)
        
        # Magic Portal Frame
        portal_rect = RoundedRectangle(corner_radius=0.2, height=3.5, width=2.5, color=GREY_B)
        # Fix for Issue 29: Move portal further right to avoid overlap with lecture text
        self.place_in_area(portal_rect, "B4", "F6", scale_factor=0.9)
        portal_label = Text("Magic Portal", font_size=18, color=GREY_A).next_to(portal_rect, UP, buff=0.1)
        
        # Input slots (n=3)
        input_slots = VGroup(*[
            Circle(radius=0.15, color=COLOR_N, fill_opacity=0.3) 
            for _ in range(3)
        ]).arrange(DOWN, buff=0.6)
        
        # Position inputs on the left edge of the portal
        input_slots.move_to(portal_rect.get_left())
        
        self.play(
            Create(portal_rect),
            Write(portal_label),
            FadeIn(input_slots, shift=RIGHT),
            dim_n.animate.scale(1.2).set_color(COLOR_N),
            run_time=1.5
        )
        self.play(dim_n.animate.scale(1/1.2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight 'm' and build the portal outputs
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_M)
        
        # Output paths (m=2)
        output_paths = VGroup(*[
            Circle(radius=0.15, color=COLOR_M, fill_opacity=0.3) 
            for _ in range(2)
        ]).arrange(DOWN, buff=1.2)
        
        # Position outputs on the right edge of the portal
        output_paths.move_to(portal_rect.get_right())
        
        self.play(
            FadeIn(output_paths, shift=RIGHT),
            dim_m.animate.scale(1.2).set_color(COLOR_M),
            run_time=1.5
        )
        self.play(dim_m.animate.scale(1/1.2))
        self.wait(1)

        # Final sequence: Vectors passing through
        # Create small arrows representing basis vectors/data
        in_arrows = VGroup(*[
            Arrow(start=LEFT*0.5, end=RIGHT*0.1, color=COLOR_N, buff=0).scale(0.5)
            for _ in range(3)
        ])
        for i, arrow in enumerate(in_arrows):
            arrow.move_to(input_slots[i].get_center() + LEFT*0.4)

        out_arrows = VGroup(*[
            Arrow(start=LEFT*0.1, end=RIGHT*0.5, color=COLOR_M, buff=0).scale(0.5)
            for _ in range(2)
        ])
        for i, arrow in enumerate(out_arrows):
            arrow.move_to(output_paths[i].get_center() + RIGHT*0.4)

        self.play(
            LaggedStart(*[
                Succession(
                    arrow.animate.move_to(input_slots[i].get_center()),
                    FadeOut(arrow, scale=0.5)
                ) for i, arrow in enumerate(in_arrows)
            ], lag_ratio=0.3)
        )
        
        self.play(
            LaggedStart(*[
                Succession(
                    FadeIn(arrow, scale=0.5),
                    arrow.animate.shift(RIGHT*0.4)
                ) for i, arrow in enumerate(out_arrows)
            ], lag_ratio=0.3)
        )

        self.wait(2)
        self.lecture[2].set_color(WHITE)
