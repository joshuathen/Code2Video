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

class Section4Scene(TeachingScene):
    def construct(self):
        # Initial Setup
        title = "Orientation: What if the Determinant is Negative?"
        lines = [
            "Negative determinants indicate the space has been flipped.",
            "It's like looking at a mirror image of space.",
            "The relative orientation of the basis vectors has reversed."
        ]
        self.setup_layout(title, lines)

        # COLORS
        COLOR_I = "#FF0000" # i-hat Red
        COLOR_J = "#0000FF" # j-hat Blue
        COLOR_TEXT = "#FF0000" # Determinant Red

        # === Animation for Lecture Line 1 ===
        # 1. Negative determinants indicate the space has been flipped.
        self.play(self.lecture[0].animate.set_color(YELLOW))

        # Setup coordinate system
        plane = NumberPlane(
            x_range=[-2, 2],
            y_range=[-2, 2],
            x_length=4,
            y_length=4,
            background_line_style={"stroke_opacity": 0.4}
        )
        self.place_in_area(plane, "B2", "E5")
        
        # Create vectors
        i_hat = Vector([1, 0], color=COLOR_I)
        j_hat = Vector([0, 1], color=COLOR_J)
        
        # Vector labels - Using Text instead of MathTex to avoid LaTeX dependency
        i_label = Text("i", color=COLOR_I, font_size=24, slant=ITALIC)
        j_label = Text("j", color=COLOR_J, font_size=24, slant=ITALIC)
        
        # Add updater-based labels to follow vectors
        i_label.add_updater(lambda m: m.next_to(i_hat.get_end(), RIGHT, buff=0.1))
        j_label.add_updater(lambda m: m.next_to(j_hat.get_end(), UP, buff=0.1))

        # Group components to move them together to the grid
        self.add(plane, i_hat, j_hat, i_label, j_label)
        self.wait(1)

        # Transform using matrix [[0, 1], [1, 0]]
        # This swaps i-hat to (0,1) and j-hat to (1,0)
        matrix = [[0, 1], [1, 0]]
        self.play(
            i_hat.animate.set_column_vector(matrix[0]),
            j_hat.animate.set_column_vector(matrix[1]),
            plane.animate.apply_matrix(matrix),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # 2. It's like looking at a mirror image of space.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )

        # Mirror Asset integration
        try:
            mirror_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/mirror.svg")
            self.place_at_grid(mirror_asset, "A4", scale_factor=0.6)
            self.play(FadeIn(mirror_asset))
            self.play(mirror_asset.animate.scale(1.2), run_time=0.5)
            self.play(mirror_asset.animate.scale(1/1.2), run_time=0.5)
            self.wait(1)
            self.play(FadeOut(mirror_asset))
        except Exception:
            # Fallback if asset is missing
            mirror_rect = Rectangle(width=0.1, height=1, color=WHITE).rotate(PI/4)
            self.place_at_grid(mirror_rect, "A4")
            self.play(Create(mirror_rect))
            self.wait(1)
            self.play(FadeOut(mirror_rect))

        # === Animation for Lecture Line 3 ===
        # 3. The relative orientation of the basis vectors has reversed.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )

        # Determinant text - Using Text instead of MathTex to avoid LaTeX dependency
        det_text = Text("det(A) = -1", color=COLOR_TEXT, font_size=36)
        self.place_at_grid(det_text, "E4")
        
        # Orientation Arrow
        orient_arrow = CurvedArrow(
            start_point=i_hat.get_end() + 0.2 * LEFT, 
            end_point=j_hat.get_end() + 0.2 * UP, 
            angle=-TAU/4, 
            color=YELLOW,
            stroke_width=3
        )
        
        self.play(Write(det_text))
        self.play(Create(orient_arrow))
        self.wait(2)

        # Reset colors for final state
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
