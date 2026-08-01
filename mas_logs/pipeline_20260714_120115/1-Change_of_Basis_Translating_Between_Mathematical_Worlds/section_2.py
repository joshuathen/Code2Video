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
        # 1. Title and lecture lines from shared state
        title_text = "Prerequisite: What is a Basis?"
        lecture_lines = [
            "Standard basis vectors i-hat and j-hat form squares.",
            "Any two non-parallel vectors can form a new grid.",
            "Let's define a new basis with two slanted vectors."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        I_COLOR = "#00FF00" # Green
        J_COLOR = "#FF0000" # Red
        B1_COLOR = "#00FFFF" # Cyan
        B2_COLOR = "#FFFF00" # Yellow
        
        # === Animation for Lecture Line 1 ===
        # Standard basis vectors i-hat and j-hat form squares.
        self.lecture[0].set_color(I_COLOR)
        
        # Visual: Standard basis (1,0) and (0,1)
        # Load the asset as required by L009 and Issue 21
        grid_asset = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/grid.svg").set_stroke(opacity=0.3)
        
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=4,
            y_length=4,
            background_line_style={"stroke_opacity": 0.4}
        )
        
        # Basis vectors and labels (using Text to avoid LaTeX errors)
        i_vec = Arrow(start=plane.c2p(0,0), end=plane.c2p(1,0), buff=0, color=I_COLOR)
        j_vec = Arrow(start=plane.c2p(0,0), end=plane.c2p(0,1), buff=0, color=J_COLOR)
        i_label = Text("i", slant=ITALIC, color=I_COLOR, font_size=24).next_to(i_vec, DOWN, buff=0.1)
        j_label = Text("j", slant=ITALIC, color=J_COLOR, font_size=24).next_to(j_vec, LEFT, buff=0.1)
        
        # Group and position in the right visual area
        # Applying fix from Issue 27: B2 to E6 with scale 0.65 to prevent bottom clipping
        viz_group = VGroup(grid_asset, plane, i_vec, j_vec, i_label, j_label)
        self.place_in_area(viz_group, 'B2', 'E6', scale_factor=0.65)
        
        self.play(FadeIn(grid_asset), FadeIn(plane))
        self.play(GrowArrow(i_vec), Write(i_label))
        self.play(GrowArrow(j_vec), Write(j_label))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # Any two non-parallel vectors can form a new grid.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(B1_COLOR)
        
        # Morph i, j into b1=(2,1) and b2=(-1,1)
        b1_target_pos = plane.c2p(2, 1)
        b2_target_pos = plane.c2p(-1, 1)
        
        b1_vec = Arrow(start=plane.c2p(0,0), end=b1_target_pos, buff=0, color=B1_COLOR)
        b2_vec = Arrow(start=plane.c2p(0,0), end=b2_target_pos, buff=0, color=B2_COLOR)
        
        b1_label = Text("b1", slant=ITALIC, color=B1_COLOR, font_size=24).next_to(b1_vec, RIGHT, buff=0.1)
        b2_label = Text("b2", slant=ITALIC, color=B2_COLOR, font_size=24).next_to(b2_vec, LEFT, buff=0.1)
        
        self.play(
            ReplacementTransform(i_vec, b1_vec),
            ReplacementTransform(i_label, b1_label),
            ReplacementTransform(j_vec, b2_vec),
            ReplacementTransform(j_label, b2_label)
        )
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        # Let's define a new basis with two slanted vectors.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(B2_COLOR)
        
        # Transform the plane to skewed
        matrix = [[2, -1], [1, 1]]
        
        skewed_plane = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=4,
            y_length=4,
            background_line_style={"stroke_opacity": 0.6}
        ).apply_matrix(matrix)
        
        # Keep same center as original plane
        skewed_plane.move_to(plane.get_center())
        
        # Morph the standard grid into the skewed coordinate system
        self.play(
            ReplacementTransform(plane, skewed_plane),
            FadeOut(grid_asset) # Removing the static grid asset as we skew
        )
        self.wait(2)
