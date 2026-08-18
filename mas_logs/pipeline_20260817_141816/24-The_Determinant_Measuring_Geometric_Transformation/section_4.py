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
        lecture_lines = ["Zero determinant means space collapses.", "Dimensions are lost in the process.", "Information is flattened completely."]
        self.setup_layout("The Zero Case: Collapsing Dimensions", lecture_lines)
        
        # Grid representation
        plane = NumberPlane(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_numbers": False}).scale(0.5)
        self.place_in_area(plane, 'B4', 'E6', scale_factor=0.9)
        self.add(plane)

        # Asset loading
        empty_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/empty.svg")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        
        target_plane = plane.copy().apply_matrix([[1, 1], [0, 0]])
        self.play(Transform(plane, target_plane), run_time=2)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF6347"))
        
        lines = VGroup(*[Line(start=plane.c2p(-2, i), end=plane.c2p(2, i)) for i in np.arange(-2, 2.5, 0.5)])
        lines.set_stroke(color="#808080", width=2)
        self.place_in_area(lines, 'B4', 'E6', scale_factor=0.9)
        self.add(empty_icon.scale(0.5).move_to(lines.get_center()))
        self.play(FadeIn(lines), FadeIn(empty_icon))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        
        zero = Text("0", font_size=48, color="#FF0000")
        self.place_at_grid(zero, 'B5', scale_factor=1.0)
        self.play(Write(zero))
        self.wait(1)
