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
        self.setup_layout("Prerequisite Refresher: The Transformation Paradigm", [
            "Square matrices reorient coordinates in the same dimension.",
            "This transformation acts as a map for vectors.",
            "Think of rotating a vector like a compass."
        ])
        
        # Define axes
        axes = Axes(x_range=[-2, 2], y_range=[-2, 2], x_length=4, y_length=4)
        grid_lines = VGroup(*[Line(axes.c2p(-2, i), axes.c2p(2, i), stroke_opacity=0.3) for i in range(-2, 3)],
                            *[Line(axes.c2p(i, -2), axes.c2p(i, 2), stroke_opacity=0.3) for i in range(-2, 3)])
        # Fix 1: Adjusted area to avoid overlap
        self.place_in_area(grid_lines, 'C2', 'F6', scale_factor=0.5)
        
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        self.place_at_grid(compass, 'A6', scale_factor=0.3)
        
        i_hat = Vector(axes.c2p(1, 0) - axes.c2p(0, 0), color="#FF4444")
        j_hat = Vector(axes.c2p(0, 1) - axes.c2p(0, 0), color="#44FF44")
        basis = VGroup(i_hat, j_hat)
        # Fix 2: Adjust position to avoid clipping
        self.place_at_grid(basis, 'D4', scale_factor=0.4)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(grid_lines), FadeIn(basis), FadeIn(compass))
        self.lecture[0].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        target_i = Vector(axes.c2p(0.707, 0.707) - axes.c2p(0, 0), color="#FF4444")
        target_j = Vector(axes.c2p(-0.707, 0.707) - axes.c2p(0, 0), color="#44FF44")
        # Apply transformation to the group that holds the basis vectors, assuming the group was placed in the area.
        # Actually, let's keep the logic consistent.
        self.play(
            ReplacementTransform(i_hat, target_i),
            ReplacementTransform(j_hat, target_j),
            grid_lines.animate.apply_matrix([[0.707, -0.707], [0.707, 0.707]])
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF00FF")
        new_compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        new_compass.rotate(PI/4)
        # Fix 3: Adjusted compass position
        self.place_at_grid(new_compass, 'D4', scale_factor=0.4)
        self.play(ReplacementTransform(compass, new_compass))
        self.wait(2)
