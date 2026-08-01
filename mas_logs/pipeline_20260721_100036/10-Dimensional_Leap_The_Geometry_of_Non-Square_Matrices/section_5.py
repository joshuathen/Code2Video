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
        title_text = "Summary & The 'Grid' Intuition"
        lecture_lines = [
            "Columns count input dimensions; rows count output dimensions.",
            "Tall matrices embed space; wide matrices project and squash.",
            "Non-square matrices are geometric bridges across dimensions."
        ]
        self.setup_layout(title_text, lecture_lines)

        def get_simple_grid(color=WHITE, size=0.6):
            g = VGroup()
            steps = 5
            for i in range(steps + 1):
                offset = (i / steps - 0.5) * size
                g.add(Line([offset, -size/2, 0], [offset, size/2, 0], stroke_width=1, color=color))
                g.add(Line([-size/2, offset, 0], [size/2, offset, 0], stroke_width=1, color=color))
            return g

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        
        # Symbolic Matrix logic
        matrix_rect = Rectangle(height=1.5, width=1.5, color=WHITE)
        
        # Rows = Output Dim
        row_label = MathTex(r"\text{Rows } (m) \to \text{Output Dim}", color=BLUE, font_size=24)
        row_lines = VGroup(*[Line(LEFT, RIGHT, color=BLUE, stroke_width=2).scale(0.4) for _ in range(3)]).arrange(DOWN, buff=0.3)
        row_lines.move_to(matrix_rect.get_center())
        row_label.next_to(matrix_rect, LEFT, buff=0.3)
        
        # Cols = Input Dim
        col_label = MathTex(r"\text{Cols } (n) \to \text{Input Dim}", color=GREEN, font_size=24)
        col_lines = VGroup(*[Line(UP, DOWN, color=GREEN, stroke_width=2).scale(0.4) for _ in range(4)]).arrange(RIGHT, buff=0.25)
        col_lines.move_to(matrix_rect.get_center())
        col_label.next_to(matrix_rect, UP, buff=0.3)

        # Grouping to move everything together
        matrix_full_group = VGroup(matrix_rect, row_lines, col_lines, row_label, col_label)
        # Fix for Issue 29
        self.place_in_area(matrix_full_group, "A3", "B6", scale_factor=0.9)

        self.play(
            Create(matrix_rect),
            Write(row_label),
            Write(col_label),
            FadeIn(row_lines),
            FadeIn(col_lines),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)

        # Tall visual: 2D -> 3D
        tall_rect = Rectangle(height=1.0, width=0.5, color="#00FF00")
        tall_label = Text("Tall (Embed)", font_size=16, color="#00FF00")
        g2d_t = get_simple_grid(color=WHITE)
        # Replaced .shear() with .apply_matrix() for Manim Community Edition
        g3d_t = get_simple_grid(color=WHITE).apply_matrix([[1, 0.3, 0], [0, 1, 0], [0, 0, 1]])
        arrow_t = Arrow(LEFT, RIGHT, color=WHITE, buff=0.1).scale(0.3)
        tall_viz = VGroup(g2d_t, arrow_t, g3d_t).arrange(RIGHT, buff=0.1)
        tall_group = VGroup(tall_label, tall_rect, tall_viz).arrange(DOWN, buff=0.1)
        # Fix for Issue 28
        self.place_in_area(tall_group, "C2", "D3", scale_factor=0.8)

        # Wide visual: 3D -> 2D
        wide_rect = Rectangle(height=0.5, width=1.0, color="#FF0000")
        wide_label = Text("Wide (Project)", font_size=16, color="#FF0000")
        # Replaced .shear() with .apply_matrix() for Manim Community Edition
        g3d_w = get_simple_grid(color=WHITE).apply_matrix([[1, 0.3, 0], [0, 1, 0], [0, 0, 1]])
        g2d_w = get_simple_grid(color=WHITE)
        arrow_w = Arrow(LEFT, RIGHT, color=WHITE, buff=0.1).scale(0.3)
        wide_viz = VGroup(g3d_w, arrow_w, g2d_w).arrange(RIGHT, buff=0.1)
        wide_group = VGroup(wide_label, wide_rect, wide_viz).arrange(DOWN, buff=0.1)
        self.place_in_area(wide_group, "C4", "D6", scale_factor=0.9)

        self.play(
            FadeIn(tall_group),
            FadeIn(wide_group),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)

        rn_tag = MathTex(r"\mathbb{R}^n", color=GREEN, font_size=28)
        rm_tag = MathTex(r"\mathbb{R}^m", color=BLUE, font_size=28)
        # Fix for Issue 30
        self.place_at_grid(rn_tag, "E2", scale_factor=0.8)
        self.place_at_grid(rm_tag, "E6", scale_factor=0.8)
        
        # Bridge Curve
        bridge_curve = CurvedArrow(self.grid["F2"], self.grid["F6"], angle=-TAU/8, color="#FFFF00")
        bridge_tag = Text("Geometric Bridge", font_size=22, color="#FFFF00")
        self.place_in_area(bridge_tag, "F3", "F5")

        self.play(
            Create(bridge_curve),
            Write(rn_tag),
            Write(rm_tag),
            FadeIn(bridge_tag),
            run_time=2
        )
        self.wait(3)
